/* gauntlet.mjs — 拉练：对上「认真版」对手还赢不赢
   node gauntlet.mjs [每场景场数] */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './harness.mjs';
import { makeHardBot } from './hardbot.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');
const N = parseInt(process.argv[2] || '12', 10);
const hard = makeHardBot();
const hardDumb = makeHardBot({ dodge: false, lead: 0.85, fireArc: 0.09 });

function scenario(label, mk, rounds = 4) {
  let roundsN = 0, wins = 0, stalls = 0, suicides = 0, deaths = 0, kills = 0, matchWins = 0;
  for (let m = 0; m < N; m++) {
    const seats = mk(m);
    const qIdx = [];
    for (let i = 0; i < 8; i++) if (seats[i] === 'qwen') qIdx.push(i);
    const s = playMatch({ seats, policyFile: POLICY, roundsToWin: rounds, seed: 777 + m * 313, dt: 1 / 60, maxMatchSeconds: 300 });
    if (s.winner !== null && qIdx.includes(s.winner)) matchWins++;
    for (const r of s.rounds) {
      if (r.unfinished) { stalls++; continue; }
      roundsN++;
      if (qIdx.includes(r.winner)) wins++;
    }
    for (const i of qIdx) { suicides += s.per[i].suicides; deaths += s.per[i].suicides + s.per[i].killedByOthers; kills += s.per[i].kills; }
  }
  const share = wins / (roundsN || 1);
  const fair = qIdxOf(mk) / (countActive(mk) || 1);
  return {
    场景: label,
    回合胜率: (share * 100).toFixed(1) + '%',
    相对公平线: fair ? (share / fair).toFixed(2) + '×' : '-',
    场胜率: (matchWins / N * 100).toFixed(0) + '%',
    击杀每回合: (kills / (roundsN || 1)).toFixed(2),
    死亡: deaths,
    自杀: suicides,
    僵局: stalls,
  };
}

function qIdxOf(mk) { const s = mk(0); let n = 0; for (let i = 0; i < 8; i++) if (s[i] === 'qwen') n++; return n; }
function countActive(mk) { const s = mk(0); let n = 0; for (let i = 0; i < 8; i++) if (s[i] !== undefined) n++; return n; }

function mkMix(qCount, hardCount, hardFn) {
  return () => {
    const seats = new Array(8).fill(undefined);
    for (let i = 0; i < qCount; i++) seats[i] = 'qwen';
    for (let i = 0; i < hardCount; i++) seats[qCount + i] = hardFn;
    return seats;
  };
}

const rows = [
  scenario('1 qwen vs 7 强敌', mkMix(1, 7, hard)),
  scenario('1 qwen vs 3 强敌 (4车)', mkMix(1, 3, hard)),
  scenario('4 qwen vs 4 强敌', mkMix(4, 4, hard)),
  scenario('1 qwen vs 7 半吊子', mkMix(1, 7, hardDumb)),
  scenario('1 qwen vs 7 内置', mkMix(1, 7, null)),
];
console.log(`\n== 拉练 · 每场景 ${N} 场（'内置' = null 座位，半吊子 = 不躲弹、提前量差）==`);
console.table(rows);
