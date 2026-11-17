import { playMatch } from './harness.mjs';

/* 全内置 bot 的冒烟测试：确认无头 Arena 跑的是真实规则 */
const seats = new Array(8).fill(null);
const t0 = Date.now();
const s = playMatch({ seats, roundsToWin: 3, seed: 7, dt: 1 / 60, maxMatchSeconds: 240 });
const ms = Date.now() - t0;

console.log('winner      :', s.winner === null ? '(未打完)' : 'P' + s.winner);
console.log('scores      :', s.scores.join(' '));
console.log('kills       :', s.kills.join(' '));
console.log('rounds      :', s.rounds.length, '| 平局', s.draws, '|', s.rounds.map((r) => `R${r.round}->P${r.winner} ${r.duration.toFixed(1)}s`).join(' '));
console.log('deaths      :', s.deaths.length, ' suicides:', s.suicides, ' 反弹杀人:', s.deaths.filter((d) => !d.suicide && d.bounces > 0).length);
console.log('sim seconds :', s.simSeconds.toFixed(1), ' wall ms:', ms, ' speed:', (s.simSeconds / (ms / 1000)).toFixed(1) + 'x');
console.table(s.per);
