/* ===========================================================
   run.mjs — 跑 208 局竞技场，统计胜率与击杀
   -----------------------------------------------------------
   用法：
     node --experimental-vm-modules arena/run.mjs
     node --experimental-vm-modules arena/run.mjs --rounds 208 --seed 20260101 --jobs 16

   开跑前会先校验策略冻结清单（policy/verify-frozen.mjs 的逻辑），
   保证战报对应的确实是冻结时的那一版代码。
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';

import { STRATEGIES, STRATEGY_IDS, MAP_SEEDS, validateRoster, RULES_VERSION } from './arena.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

/* ---------------- 参数 ---------------- */

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i >= 0 && process.argv[i + 1] != null) return process.argv[i + 1];
  return dflt;
}
const ROUNDS = parseInt(arg('rounds', '208'), 10);
const SEED = parseInt(arg('seed', '20260101'), 10);
const JOBS = parseInt(arg('jobs', String(Math.min(16, os.cpus().length))), 10);
const PLAYERS = validateRoster(arg('players', STRATEGY_IDS.join(',')).split(',').map(s => s.trim()));
const PARTICIPANTS = STRATEGIES.filter(s => PLAYERS.includes(s.id));
if (![ROUNDS, JOBS].every(n => Number.isInteger(n) && n > 0) || !Number.isFinite(SEED)) throw new Error('rounds/jobs 必须为正整数，seed 必须为数字');
const OUT = path.resolve(ROOT, arg('out', path.join('arena', 'report.json')));

/* ---------------- 冻结校验 ---------------- */

async function verifyFrozen() {
  const manifestPath = path.join(ROOT, 'policy', 'manifest.sha256.json');
  if (!fs.existsSync(manifestPath)) {
    console.error('✗ 缺少 policy/manifest.sha256.json —— 请先运行 node policy/freeze.mjs');
    process.exit(1);
  }
  const { verifyManifest } = await import(pathToFileURL(path.join(ROOT, 'policy', 'freeze.mjs')).href);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const problems = verifyManifest(manifest);
  if (problems.length) {
    console.error('✗ 策略冻结校验失败（策略被改动过），中止：');
    for (const p of problems) console.error('  - ' + p);
    process.exit(1);
  }
  const n = Object.values(manifest.strategies).reduce((a, s) => a + s.fileCount, 0);
  console.log(`✓ 策略冻结校验通过（${Object.keys(manifest.strategies).length} 个策略 / ${n} 个文件）`);
}

/* ---------------- 并行调度 ---------------- */

function runWorkers(rounds, seed, jobs) {
  return new Promise((resolve, reject) => {
    const n = Math.max(1, Math.min(jobs, rounds));
    const chunk = Math.ceil(rounds / n);
    const results = [];
    let finished = 0;
    let doneCount = 0;
    const t0 = Date.now();

    const drawProgress = () => {
      const pct = ((doneCount / rounds) * 100).toFixed(0);
      const el = ((Date.now() - t0) / 1000).toFixed(0);
      process.stdout.write(`\r  进度 ${doneCount}/${rounds} (${pct}%)  已用 ${el}s        `);
    };

    for (let i = 0; i < n; i++) {
      const from = i * chunk;
      const to = Math.min(rounds, from + chunk);
      if (from >= to) { finished++; continue; }

      const w = new Worker(path.join(HERE, 'worker.mjs'), {
        workerData: { baseSeed: seed, from, to, players: PLAYERS },
        execArgv: ['--experimental-vm-modules'],
      });

      w.on('message', (msg) => {
        if (msg.type === 'progress') {
          doneCount += msg.done;
          drawProgress();
        } else if (msg.type === 'done') {
          results.push(...msg.results);
        }
      });

      w.on('error', reject);
      w.on('exit', () => {
        finished++;
        if (finished === n) {
          process.stdout.write('\n');
          resolve(results);
        }
      });

      w.postMessage({ type: 'run' });
    }
  });
}

/* ---------------- 统计 ---------------- */

function aggregate(results) {
  const rows = {};
  for (const st of PARTICIPANTS) {
    rows[st.id] = {
      id: st.id, label: st.label,
      wins: 0, draws: 0, timeouts: 0,
      kills: 0, suicides: 0, deaths: 0, errors: 0,
      rounds: 0, aliveSum: 0, finalists: 0, finalsWins: 0,
      lifeSum: 0, lifeCount: 0,
      byMap: { A: { wins: 0, rounds: 0 }, B: { wins: 0, rounds: 0 } },
    };
  }

  for (const r of results) {
    if (r.error) { console.error('  ! 第 ' + (r.round + 1) + ' 局出错：' + r.error); continue; }
    const mapKey = r.mapSeed === MAP_SEEDS[0] ? 'A' : 'B';

    for (const id of PLAYERS) {
      const row = rows[id];
      row.rounds++;
      row.byMap[mapKey].rounds++;
      row.kills += r.kills[id] || 0;
      row.suicides += r.suicides[id] || 0;
      row.deaths += r.deaths[id] || 0;
      row.errors += r.errors[id] || 0;
      if ((r.aliveAtEnd || []).includes(id)) row.aliveSum++;
      /* 存活时长：存活到最后的用整局时长，阵亡的用死亡时刻 */
      const dt = r.deathTime && r.deathTime[id] != null ? r.deathTime[id] : r.durationSec;
      row.lifeSum += dt;
      row.lifeCount++;
    }
    if (r.winner) {
      rows[r.winner].wins++;
      rows[r.winner].byMap[mapKey].wins++;
    } else {
      for (const id of PLAYERS) rows[id].draws++;
    }
    if (r.timeout) for (const id of PLAYERS) rows[id].timeouts++;

    /* 决赛（活到最后两人） */
    for (const id of (r.finalists || [])) rows[id].finalists++;
  }

  for (const id of PLAYERS) {
    const row = rows[id];
    row.winRate = row.rounds ? row.wins / row.rounds : 0;
    row.killPerRound = row.rounds ? row.kills / row.rounds : 0;
    row.netKills = row.kills - row.suicides;
    row.netPerRound = row.rounds ? row.netKills / row.rounds : 0;
    row.deathPerRound = row.rounds ? row.deaths / row.rounds : 0;
    row.suicidePerRound = row.rounds ? row.suicides / row.rounds : 0;
    row.surviveRate = row.rounds ? row.aliveSum / row.rounds : 0;
    row.finalRate = row.rounds ? row.finalists / row.rounds : 0;
    row.finalWinRate = row.finalists ? row.wins / row.finalists : 0;
    row.avgLifeSec = row.lifeCount ? row.lifeSum / row.lifeCount : 0;
    row.mapAWinRate = row.byMap.A.rounds ? row.byMap.A.wins / row.byMap.A.rounds : 0;
    row.mapBWinRate = row.byMap.B.rounds ? row.byMap.B.wins / row.byMap.B.rounds : 0;
  }

  return rows;
}

/* ---------------- 输出 ---------------- */

function printTable(rows) {
  const list = Object.values(rows).sort((a, b) => b.winRate - a.winRate);
  const head = ['策略', '胜场', '胜率', '击杀', '场均击杀', '自杀', '净击杀', '死亡', '决赛率', '决赛胜率', '存活时长', '异常'];
  const w = [11, 5, 7, 5, 8, 5, 6, 5, 7, 8, 8, 5];
  const pad = (s, i) => {
    const str = String(s);
    const width = w[i];
    let len = 0;
    for (const ch of str) len += /[一-鿿　-〿＀-￯]/.test(ch) ? 2 : 1;
    const diff = width - len;
    if (i === 0 || i === 1) return str + ' '.repeat(Math.max(0, diff));
    return ' '.repeat(Math.max(0, diff)) + str;
  };
  console.log('\n' + head.map((h, i) => pad(h, i)).join(' '));
  console.log('─'.repeat(w.reduce((a, b) => a + b, 0) + head.length - 1));
  for (const r of list) {
    console.log([
      r.label,
      r.wins,
      (r.winRate * 100).toFixed(1) + '%',
      r.kills,
      r.killPerRound.toFixed(2),
      r.suicides,
      r.netKills,
      r.deaths,
      (r.finalRate * 100).toFixed(0) + '%',
      (r.finalRate > 0 ? (r.finalWinRate * 100).toFixed(0) + '%' : '—'),
      r.avgLifeSec.toFixed(1) + 's',
      r.errors,
    ].map((v, i) => pad(v, i)).join(' '));
  }
  console.log('\n  胜率   = 回合胜场 / ' + (list[0] ? list[0].rounds : 0) + ' 局（最后存活者获胜）');
  console.log('  击杀   = 开枪命中他人；自杀 = 自己的子弹打死自己；净击杀 = 击杀 − 自杀');
  console.log('  决赛率 = 活到本局最后两人；决赛胜率 = 进入决赛后最终夺冠的比例');
  console.log('  存活时长 = 平均在场上存活到阵亡/结束的秒数');
}

/* ---------------- 主流程 ---------------- */

await verifyFrozen();

console.log(`\n▶ Tank Trouble 竞技场：${ROUNDS} 局 · ${PLAYERS.length} 车（${PLAYERS.join(", ")}）· seed=${SEED} · jobs=${JOBS}\n`);

if (ROUNDS % (2 * PLAYERS.length)) console.log('提示：局数取 ' + (2 * PLAYERS.length) + ' 的倍数可完整配平两张地图和出生席位。');
const t0 = Date.now();
const results = await runWorkers(ROUNDS, SEED, JOBS);
results.sort((a, b) => a.round - b.round);

const rows = aggregate(results);
printTable(rows);

const timeouts = results.filter((r) => r.timeout).length;
const draws = results.filter((r) => !r.winner).length;
const avgDur = results.reduce((a, r) => a + (r.durationSec || 0), 0) / (results.length || 1);
const errRounds = results.filter((r) => r.error).length;

console.log('\n  局统计：平均时长 ' + avgDur.toFixed(1) + 's · 平局/超时 ' + draws + ' 局' +
  (timeouts ? '（其中超时 ' + timeouts + ' 局）' : '') +
  ' · 异常局 ' + errRounds + ' · 总耗时 ' + ((Date.now() - t0) / 1000).toFixed(1) + 's');

const report = {
  rulesVersion: RULES_VERSION,
  roundTimeLimitSec: 60,
  generatedAt: new Date().toISOString(),
  rounds: ROUNDS,
  seed: SEED,
  avgDurationSec: +avgDur.toFixed(2),
  timeouts,
  draws,
  errorRounds: errRounds,
  strategies: PARTICIPANTS.map((s) => s.id),
  labels: Object.fromEntries(PARTICIPANTS.map((s) => [s.id, s.label])),
  rows,
  rounds_detail: results.map((r) => ({
    round: r.round + 1, winner: r.winner, timeout: r.timeout,
    durationSec: r.durationSec, mapSeed: r.mapSeed,
    spawnSeed: r.spawnSeed, seats: r.seats, aliveAtEnd: r.aliveAtEnd, finalists: r.finalists,
    kills: r.kills, suicides: r.suicides,
    eliminationOrder: r.eliminationOrder,
  })),
};
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(report, null, 2));
console.log('\n  战报已写入 ' + path.relative(ROOT, OUT));
