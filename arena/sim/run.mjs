/* ===========================================================
   sim/run.mjs — 大规模模拟调度器
   -----------------------------------------------------------
   用法：
     node --experimental-vm-modules arena/sim/run.mjs --budget pilot
     node --experimental-vm-modules arena/sim/run.mjs --budget standard --jobs 20
     node --experimental-vm-modules arena/sim/run.mjs --stages main8 --budget quick
     node --experimental-vm-modules arena/sim/run.mjs --budget standard --fresh   # 丢弃旧结果

   特性：
     · 动态任务队列：谁先跑完谁接下一局，不吃「长局拖尾」的亏；
     · 逐局 append 到 rounds.jsonl，Ctrl-C 或崩溃都不丢已完成的对局；
     · 断点续跑：重跑同一条命令会自动跳过已有 (job, round)；
     · 任务按轮次交错派发，中途停下时每个配置都有近似等量的样本。
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';

import { buildPlan, planSummary, EFFECTIVE_CORES } from './plan.mjs';
import { RULES_VERSION } from '../arena.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');

const arg = (n, d) => {
  const i = process.argv.indexOf('--' + n);
  return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : d;
};
const flag = (n) => process.argv.includes('--' + n);

const BUDGET = arg('budget', 'standard');
const SEED = parseInt(arg('seed', '20260101'), 10);
const JOBS = Math.max(1, parseInt(arg('jobs', String(Math.max(2, Math.min(20, os.cpus().length - 4)))), 10));
const STAGES = arg('stages', 'main8,count,duel').split(',').map((s) => s.trim()).filter(Boolean);
const IDS = arg('ids', '').split(',').map((s) => s.trim()).filter(Boolean);
const OUT = path.resolve(ROOT, arg('out', path.join('output', 'sim')));
const FRESH = flag('fresh');

const plan = buildPlan({ budget: BUDGET, seed: SEED, stages: STAGES, ...(IDS.length ? { ids: IDS } : {}) });

/* ---------------- 输出位置 ---------------- */

fs.mkdirSync(OUT, { recursive: true });
const JSONL = path.join(OUT, 'rounds.jsonl');
if (FRESH && fs.existsSync(JSONL)) fs.rmSync(JSONL);
const META = path.join(OUT, 'plan.json');
if (!FRESH && fs.existsSync(JSONL) && fs.statSync(JSONL).size > 0) {
  const previous = fs.existsSync(META) ? JSON.parse(fs.readFileSync(META, 'utf8')) : {};
  if (previous.rulesVersion !== RULES_VERSION) {
    throw new Error('该输出目录包含旧规则对局，不能与 60 秒对局混合续跑。请用 --out 指定新目录，保留历史结果。');
  }
}
fs.writeFileSync(META, JSON.stringify({ rulesVersion: RULES_VERSION, roundTimeLimitSec: 60, budget: BUDGET, seed: SEED, jobs: plan.jobs, cpuSeconds: plan.cpuSeconds }, null, 2));

/* ---------------- 断点续跑 ---------------- */

const done = new Set();
if (fs.existsSync(JSONL)) {
  for (const line of fs.readFileSync(JSONL, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { done.add(JSON.parse(line).k); } catch { /* 半行忽略 */ }
  }
}

/* 任务按「轮次」交错：先所有配置的第 0 局，再第 1 局…… */
const tasks = [];
for (const job of plan.jobs) {
  for (let r = 0; r < job.rounds; r++) tasks.push({ job, index: r });
}
tasks.sort((a, b) => (a.index - b.index) || a.job.id.localeCompare(b.job.id));
const queue = tasks.filter((t) => !done.has(t.job.id + '#' + t.index));

const total = tasks.length;
let doneCount = done.size;
let failed = 0;

console.log(`\n▶ 模拟计划 ${BUDGET} · ${plan.jobs.length} 个配置 · ${total} 局 · seed=${SEED} · 并发 ${JOBS}`);
for (const [tag, s] of Object.entries(planSummary(plan))) {
  console.log(`   ${tag.padEnd(7)} ${String(s.jobs).padStart(3)} 配置 / ${String(s.rounds).padStart(5)} 局 / 预计 ${(s.seconds / 3600).toFixed(2)} 核·小时`);
}
console.log(`   合计预计 ${(plan.cpuSeconds / EFFECTIVE_CORES / 3600).toFixed(2)} 小时（${JOBS} 并发，实测聚合吞吐 ≈ ${EFFECTIVE_CORES} 核）`);
if (done.size) console.log(`   续跑：已有 ${done.size} 局，本次还需 ${queue.length} 局\n`);
else console.log('');

if (!queue.length) {
  console.log('✓ 计划已全部完成，直接运行 analyze.mjs 出表。');
  process.exit(0);
}

/* ---------------- 调度 ---------------- */

const t0 = Date.now();
let next = 0;
let inflight = 0;
let stopping = false;
const workers = new Set();

const out = fs.createWriteStream(JSONL, { flags: 'a' });

function report() {
  const elapsed = (Date.now() - t0) / 1000;
  const rate = (doneCount - done.size) / Math.max(1e-6, elapsed);
  const remain = queue.length - (doneCount - done.size);
  const eta = rate > 0 ? remain / rate : Infinity;
  const pct = ((doneCount / total) * 100).toFixed(1);
  const s = (x) => (Number.isFinite(x) ? Math.round(x) + 's' : '—');
  process.stdout.write(
    `\r   ${doneCount}/${total} (${pct}%) · ${rate.toFixed(2)} 局/s · 已用 ${s(elapsed)} · 剩余 ${s(eta)}` +
    (failed ? ` · 失败 ${failed}` : '') + '        '
  );
}

function dispatch(w) {
  if (stopping || next >= queue.length) {
    w.postMessage({ type: 'stop' });
    if (inflight === 0) finish();
    return;
  }
  const t = queue[next++];
  inflight++;
  w.postMessage({ type: 'round', job: t.job, index: t.index });
}

function spawnWorker() {
  const w = new Worker(path.join(HERE, 'worker.mjs'), { execArgv: ['--experimental-vm-modules'] });
  workers.add(w);

  w.on('message', (msg) => {
    if (msg.type !== 'result') return;
    inflight--;
    const rec = msg.res;
    out.write(JSON.stringify({ k: msg.key, ...rec }) + '\n');
    doneCount++;
    if (rec.error) failed++;
    if (doneCount % 10 === 0) report();
    dispatch(w);
  });

  w.on('error', (e) => {
    console.error('\n  ! worker 异常：' + (e && e.message));
    workers.delete(w);
    if (!stopping) spawnWorker();
  });

  w.on('exit', () => {
    workers.delete(w);
  });

  dispatch(w);
}

function finish() {
  if (finish.done) return;
  finish.done = true;
  report();
  console.log('\n\n  本次完成 ' + (doneCount - done.size) + ' 局' + (failed ? `（失败 ${failed}）` : ''));
  out.end(() => process.exit(0));
}

process.on('SIGINT', () => {
  if (stopping) process.exit(1);
  stopping = true;
  console.log('\n  收到中断：等待在跑的局收尾，已完成的对局已落盘……');
  setTimeout(() => { out.end(() => process.exit(0)); }, 1500);
});

for (let i = 0; i < Math.min(JOBS, queue.length); i++) spawnWorker();
