/* robust.mjs — 鲁棒性体检：不同地图 / 人数 / 帧率 / 镜像自战
   node robust.mjs [每场景场数] */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');
const N = parseInt(process.argv[2] || '12', 10);

const MAP_A = [20260101], MAP_B = [88888888];

function roster(spec) {
  const r = new Array(8).fill(undefined);
  const parts = spec.split('/');           // "qwen1/builtin7"
  let i = 0;
  for (const p of parts) {
    const [name, cnt] = p.split(/(\d+)/).filter(Boolean);
    const c = parseInt(cnt || '1', 10);
    for (let k = 0; k < c; k++) { r[i] = name === 'qwen' ? 'qwen' : (name === 'off' ? undefined : null); i++; }
  }
  return r;
}

function run(label, spec, opts = {}) {
  const seats = roster(spec);
  let qIdx = [];
  for (let i = 0; i < 8; i++) if (seats[i] === 'qwen') qIdx.push(i);
  let rounds = 0, wins = 0, stalls = 0, stallSec = 0, suicides = 0, deaths = 0, kills = 0, matchWins = 0, matches = 0;
  const perSeatWins = new Array(8).fill(0);
  let maxRound = 0;

  for (let m = 0; m < N; m++) {
    const s = playMatch({
      seats: seats.slice(), policyFile: POLICY, roundsToWin: opts.rounds || 4,
      seed: 4242 + m * 811, dt: opts.dt || 1 / 60, mapSeeds: opts.maps || null,
      maxMatchSeconds: opts.cap || 300,
    });
    matches++;
    if (s.winner !== null && qIdx.includes(s.winner)) matchWins++;
    for (const r of s.rounds) {
      if (r.unfinished) { stalls++; stallSec += r.duration; continue; }
      rounds++;
      if (r.duration > maxRound) maxRound = r.duration;
      if (qIdx.includes(r.winner)) wins++;
      if (r.winner >= 0) perSeatWins[r.winner]++;
    }
    for (const i of qIdx) { suicides += s.per[i].suicides; deaths += s.per[i].suicides + s.per[i].killedByOthers; kills += s.per[i].kills; }
  }

  const share = wins / (rounds || 1);
  const fair = qIdx.length / (seats.filter((x) => x !== undefined).length || 1);
  return {
    场景: label,
    座位: qIdx.length + '/' + seats.filter((x) => x !== undefined).length,
    回合胜率: (share * 100).toFixed(1) + '%',
    相对公平线: share > 0 ? (share / fair).toFixed(2) + '×' : '-',
    场胜率: (matchWins / matches * 100).toFixed(0) + '%',
    击杀每回合: (kills / (rounds || 1)).toFixed(2),
    自杀: suicides,
    僵死回合: stalls,
    最长回合: maxRound.toFixed(0) + 's',
  };
}

const rows = [];
rows.push(run('常规 1v7 · 双图', 'qwen1/builtin7'));
rows.push(run('仅地图 A', 'qwen1/builtin7', { maps: MAP_A }));
rows.push(run('仅地图 B', 'qwen1/builtin7', { maps: MAP_B }));
rows.push(run('4 车局 1v3', 'qwen1/builtin3/off4'));
rows.push(run('6 车局 1v5', 'qwen1/builtin5/off2'));
rows.push(run('2 席 qwen + 6 内置', 'qwen2/builtin6'));
rows.push(run('镜像 8 qwen 自战', 'qwen8'));
rows.push(run('低帧 dt=1/30', 'qwen1/builtin7', { dt: 1 / 30 }));
rows.push(run('很低帧 dt=1/15', 'qwen1/builtin7', { dt: 1 / 15 }));
rows.push(run('高帧 dt=1/120', 'qwen1/builtin7', { dt: 1 / 120 }));

console.log(`\n== 鲁棒性体检 · 每场景 ${N} 场 ==`);
console.table(rows);
