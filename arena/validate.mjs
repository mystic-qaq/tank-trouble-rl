/* ===========================================================
   validate.mjs — 竞技场集成校验
   -----------------------------------------------------------
   目的：验证「竞技场对每个策略的驱动方式」与策略自带的测试台一致。

   做法：每个策略单独上场，对手全是引擎内置 Bot（1 vs 7），
   跑若干局看回合胜率，跟各策略 README 自测的数字对��。

   如果某个策略在这里明显低于它自己的自测成绩，说明是竞技场
   适配出了问题（而不是策略本身弱），需要回去修接口。

   用法：node --experimental-vm-modules arena/validate.mjs --rounds 40
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';

import { STRATEGIES, STRATEGY_IDS } from './arena.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i >= 0 && process.argv[i + 1] != null) return process.argv[i + 1];
  return dflt;
}
const ROUNDS = parseInt(arg('rounds', '40'), 10);
const SEED = parseInt(arg('seed', '20260101'), 10);
const JOBS = parseInt(arg('jobs', String(Math.min(16, os.cpus().length))), 10);

/* 各策略 README 自测的「1 席 vs 7 内置 Bot」回合胜率 */
const SELF_REPORTED = {
  deepseek: '44.7%',
  glm: '~52%',
  gpt: '（README 未给 1v7 胜率）',
  hy: '54.0%',
  kimi: '（README 未给 1v7 胜率）',
  minimax: '95.0%',
  qwen: '40.0%',
  builtin: '~14%',
};

/* ---------------- 并行 ---------------- */

function runAll(baseSeed, rounds, jobs) {
  return new Promise((resolve, reject) => {
    const tasks = [];
    for (const st of STRATEGIES) {
      if (st.id === 'builtin') continue;
      for (let r = 0; r < rounds; r++) tasks.push({ subject: st.id, round: r });
    }
    const n = Math.max(1, Math.min(jobs, tasks.length));
    const chunk = Math.ceil(tasks.length / n);
    const results = [];
    let finished = 0;

    for (let i = 0; i < n; i++) {
      const slice = tasks.slice(i * chunk, (i + 1) * chunk);
      if (!slice.length) { finished++; continue; }
      const w = new Worker(path.join(HERE, 'validate-worker.mjs'), {
        workerData: { baseSeed, slice },
        execArgv: ['--experimental-vm-modules'],
      });
      w.on('message', (m) => { if (m.type === 'done') results.push(...m.results); });
      w.on('error', reject);
      w.on('exit', () => { if (++finished === n) resolve(results); });
      w.postMessage({ type: 'run' });
    }
  });
}

/* ---------------- 主流程 ---------------- */

const results = await runAll(SEED, ROUNDS, JOBS);
console.log(`\n▶ 集成校验：每个策略单独上场 vs 7 个内置 Bot，${ROUNDS} 局/策略\n`);
console.log('  策略          本次胜率    自测胜率(README)   平均存活时长   结论');
console.log('  ' + '─'.repeat(74));

for (const st of STRATEGIES) {
  if (st.id === 'builtin') continue;
  const mine = results.filter((r) => r.subject === st.id);
  const wins = mine.filter((r) => r.winner === st.id).length;
  const rate = mine.length ? wins / mine.length : 0;
  const avgLife = mine.length ? mine.reduce((a, r) => a + r.durationSec, 0) / mine.length : 0;
  const pct = (rate * 100).toFixed(1) + '%';
  const self = SELF_REPORTED[st.id] || '—';
  let verdict;
  if (rate >= 0.35) verdict = '✓ 与自测相符';
  else if (rate >= 0.20) verdict = '△ 低于自测，需核查';
  else verdict = '✗ 明显异常，需核查接口';
  console.log(
    '  ' + st.label.padEnd(14) + pct.padStart(8) + '    ' +
    String(self).padEnd(22) + (avgLife.toFixed(1) + 's').padStart(10) + '     ' + verdict
  );
}
console.log('');
