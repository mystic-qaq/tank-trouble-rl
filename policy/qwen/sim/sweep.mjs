/* sweep.mjs — 并行调参台
   node sweep.mjs configs.json [--n 40] [--seats 1] [--jobs 6]
   configs.json: [ { "label": "base", "cfg": {} }, { "label": "m10", "cfg": {"selfMargin":10} } ] */
import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RUNNER = path.join(HERE, 'runner.mjs');

function arg(n, d) {
  const i = process.argv.indexOf('--' + n);
  return i < 0 ? d : process.argv[i + 1];
}

const file = process.argv[2];
if (!file || file.startsWith('--')) {
  console.error('用法: node sweep.mjs configs.json [--n 40] [--seats 1] [--jobs 6] [--seed 1000]');
  process.exit(1);
}
const configs = JSON.parse(fs.readFileSync(file, 'utf8'));
const N = arg('n', '40');
const SEATS = arg('seats', '1');
const JOBS = parseInt(arg('jobs', '6'), 10);
const SEED = arg('seed', '1000');
const OPP = arg('opponent', 'builtin');
const DT = arg('dt', String(1 / 60));

function run(cfg) {
  return new Promise((resolve) => {
    execFile(process.execPath, [RUNNER, '--n', N, '--seats', SEATS, '--seed', SEED,
      '--cfg', JSON.stringify(cfg.cfg || {}), '--opponent', OPP, '--dt', DT],
      { maxBuffer: 1 << 22 }, (err, stdout) => {
        if (err) { resolve({ error: String(err).slice(0, 200), cfg }); return; }
        try { resolve(JSON.parse(stdout.trim().split('\n').pop())); }
        catch (e) { resolve({ error: 'parse: ' + stdout.slice(0, 200) }); }
      });
  });
}

const results = new Array(configs.length);
let next = 0;
async function worker() {
  while (true) {
    const i = next++;
    if (i >= configs.length) return;
    const label = configs[i].label || ('#' + i);
    const res = await run(configs[i]);
    res.label = label;
    results[i] = res;
    process.stderr.write('.');
  }
}
await Promise.all(Array.from({ length: Math.min(JOBS, configs.length) }, worker));

const base = results[0];
const pct = (x) => (x * 100).toFixed(1) + '%';
console.log(`\n== sweep ==  ${N} 场/配置 · 每配置 ${SEATS} 个 qwen 座位 · 对手 ${OPP}`);
console.log('（回合胜率 = 我作为唯一幸存者的比率，等于生存率）\n');
const rows = results.map((r) => {
  if (!r || r.error) return { 配置: configs[results.indexOf(r)]?.label || '?', 错误: (r && r.error) || 'null' };
  const se = Math.sqrt((r.winRate * (1 - r.winRate)) / Math.max(1, r.rounds));
  const d = r.winRate - base.winRate;
  const dse = base.winRate != null ? Math.sqrt(se * se + (Math.sqrt(base.winRate * (1 - base.winRate) / base.rounds)) ** 2) : 0;
  return {
    配置: r.label,
    回合胜率: pct(r.winRate),
    'Δ vs base': (d >= 0 ? '+' : '') + (d * 100).toFixed(1) + '%',
    显著性: Math.abs(d / dse).toFixed(2) + 'σ',
    场胜率: pct(r.matchWin),
    单挑转化: r.duoWinRate === null ? '-' : pct(r.duoWinRate),
    僵局场: r.stalls + '/' + r.n,
    僵死均秒: r.stallSec.toFixed(0),
    自杀: r.suicideRate.toFixed(3),
    击杀: r.killRate.toFixed(2),
    '均回合s': r.avgRound.toFixed(1),
    cfg: JSON.stringify(r.cfg),
  };
});
console.table(rows);
