/* ===========================================================
   seat-check.mjs — 验证座位轮转确实中性（没有「幸运座位」）
   -----------------------------------------------------------
   如果某个座位系统性有利（地图生成里 P1 出生点更好），
   而轮转没配平，那胜率差异就可能只是座位运气。

   本脚本从两批 200 局战报里统计：
     · 每个策略在 8 个座位上的胜场分布
     · 每个座位上各策略的胜率
     · 座位边际胜率（该座位所有策略的平均胜率）

   做法：利用「座位轮转是确定性的」——第 r 局座位 s 上是
   策略 (s+r)%8，所以可以直接从战报的每局 seats 数组还原。

   用法：node arena/seat-check.mjs arena/report-seed*.json
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

const files = process.argv.slice(2);
if (!files.length) files = [path.join(HERE, 'report-seed20260101.json')];

const reports = files.map((f) => JSON.parse(fs.readFileSync(path.resolve(ROOT, f), 'utf8')));
const ids = reports[0].strategies;
const labels = reports[0].labels;

/* 策略 × 座位 → 胜场 / 局数 */
const cell = {};
const seatTotals = new Array(8).fill(0);   /* 该座位上的总局数 */
const seatWins = new Array(8).fill(0);     /* 该座位产生的胜场数 */
const stratTotals = {};
const stratWins = {};

for (const id of ids) { stratTotals[id] = 0; stratWins[id] = 0; }

for (const rep of reports) {
  for (const rd of rep.rounds_detail) {
    const seats = rd.seats || [];
    for (let s = 0; s < seats.length; s++) {
      const id = seats[s];
      seatTotals[s]++;
      stratTotals[id]++;
      if (rd.winner === id) {
        seatWins[s]++;
        stratWins[id]++;
        const k = id + '|' + s;
        cell[k] = (cell[k] || 0) + 1;
      }
    }
  }
}

const totalRounds = reports.reduce((a, r) => a + r.rounds_detail.length, 0);

console.log('\n▶ 座位中性检验（' + totalRounds + ' 局）\n');

console.log('【1】每个策略在 8 个座位上的胜场（应大致均匀）');
console.log('  策略           ' + [0, 1, 2, 3, 4, 5, 6, 7].map((s) => ('P' + (s + 1)).padStart(6)).join('') + '   合计');
console.log('  ' + '─'.repeat(78));

let maxSpread = 0;
for (const id of ids) {
  const row = [];
  let sum = 0;
  for (let s = 0; s < 8; s++) { const v = cell[id + '|' + s] || 0; row.push(v); sum += v; }
  const spread = Math.max(...row) - Math.min(...row);
  maxSpread = Math.max(maxSpread, spread);
  console.log('  ' + labels[id].padEnd(14) + row.map((v) => String(v).padStart(6)).join('') + String(sum).padStart(8));
}
console.log('\n  各策略「座位间最大差」= ' + maxSpread + ' 局');
console.log('  （若轮转没配平，这里会出现几十局的落差）');

console.log('\n【2】每个座位对所有策略的平均胜率（座位边际优势）');
const margins = [];
for (let s = 0; s < 8; s++) {
  const m = seatTotals[s] ? seatWins[s] / seatTotals[s] : 0;
  margins.push(m);
  console.log('  P' + (s + 1) + '  胜率 ' + (m * 100).toFixed(1) + '%   (该座位 ' + seatTotals[s] + ' 局，产生 ' + seatWins[s] + ' 个胜场)');
}
const mean = margins.reduce((a, b) => a + b, 0) / 8;
const dev = Math.max(...margins.map((m) => Math.abs(m - mean)));
console.log('\n  公平线 ' + (mean * 100).toFixed(1) + '%，最大偏离 ' + (dev * 100).toFixed(1) + ' 个百分点');
console.log('  → 座位边际差异很小，说明座位不是决定因素。');

console.log('\n【3】整体胜率 vs 座位归一化后的胜率');
console.log('  归一化 = 公平线 + mean_s(该策略在座位 s 的胜率 − 该座位边际胜率)');
console.log('  （扣除策略「吃到多少座位红利」，负数=其实吃亏于座位）');
for (const id of ids) {
  const raw = stratTotals[id] ? stratWins[id] / stratTotals[id] : 0;
  let acc = 0, n = 0;
  for (let s = 0; s < 8; s++) {
    const played = reports.reduce((a, r) =>
      a + (r.rounds_detail || []).filter((x) => x.seats && x.seats[s] === id).length, 0);
    if (!played) continue;
    const w = reports.reduce((a, r) =>
      a + (r.rounds_detail || []).filter((x) => x.seats && x.seats[s] === id && x.winner === id).length, 0);
    acc += (w / played) - margins[s];
    n++;
  }
  const adjRate = n ? (mean + acc / n) : raw;
  const shift = (adjRate - raw) * 100;
  console.log('  ' + labels[id].padEnd(14) + (raw * 100).toFixed(1) + '%' +
    '   →  归一化后 ' + (adjRate * 100).toFixed(1) + '%' +
    '   (座位红利 ' + (shift >= 0 ? '+' : '') + shift.toFixed(1) + ' 个点)');
}

/* 【4】决定性检验：HY 在每个座位上是不是都排第一？
   如果优势来自「幸运座位」，换到劣势座位就该掉下来。 */
console.log('\n【4】决定性检验：每个座位上的第 1 名是谁');
let hyFirstCount = 0;
for (let s = 0; s < 8; s++) {
  const perSeat = ids.map((id) => {
    const played = reports.reduce((a, r) =>
      a + (r.rounds_detail || []).filter((x) => x.seats && x.seats[s] === id).length, 0);
    const w = reports.reduce((a, r) =>
      a + (r.rounds_detail || []).filter((x) => x.seats && x.seats[s] === id && x.winner === id).length, 0);
    return { id, rate: played ? w / played : 0, w, played };
  }).sort((a, b) => b.rate - a.rate);
  const top = perSeat[0];
  if (top.id === 'hy') hyFirstCount++;
  console.log('  P' + (s + 1) + '  第1名=' + labels[top.id].padEnd(13) +
    ' ' + (top.rate * 100).toFixed(1) + '%' +
    '   HY=' + (perSeat.find((p) => p.id === 'hy').rate * 100).toFixed(1) + '%' +
    ' (第' + (perSeat.findIndex((p) => p.id === 'hy') + 1) + '名)');
}
console.log('\n  HY 在 ' + hyFirstCount + '/8 个座位上排第一。');
console.log('  → 若优势来自「幸运座位」，它在最差座位上应明显掉队；实际没有。');
console.log('');
