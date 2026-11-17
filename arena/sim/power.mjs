/* ===========================================================
   sim/power.mjs — 样本量计算器
   -----------------------------------------------------------
   「要跑多少局」不是一个拍脑袋的数，取决于三件事：
     1) 你要的分辨率（胜率误差 ±几pp、能看出两策略差几pp）；
     2) 一局里有几个人（N 人混战时基线胜率 = 1/N，方差最大点在 0.5，不是 1/N）；
     3) 同一套出生布局下的对局是相关的，有效样本 < 名义局数（设计效应 DEFF）。

   用法：
     node arena/sim/power.mjs
     node arena/sim/power.mjs --deff 1.2 --n 8
   =========================================================== */

const arg = (n, d) => {
  const i = process.argv.indexOf('--' + n);
  return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : d;
};

const Z = 1.959964;       // 95% 双侧
const ZP = 0.8416212;     // 80% power
const DEFF = parseFloat(arg('deff', '1.15'));

/* 单个策略胜率的 95% CI 半宽 = Z*sqrt(p(1-p)*DEFF/R) → 反解 R */
const roundsForHalfWidth = (p, h, deff = DEFF) => Math.ceil((Z * Z * p * (1 - p) * deff) / (h * h));

/* 同场两策略胜率之差的方差：Var(X-Y)=p1(1-p1)+p2(1-p2)+2p1p2（一局只有一个赢家 → 负相关） */
const diffVar = (p1, p2) => p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2;

/* 要在 80% power 下分辨 Δ 的差距，需要多少局 */
const roundsForDiff = (p1, p2, delta, deff = DEFF) =>
  Math.ceil(((Z + ZP) ** 2 * diffVar(p1, p2) * deff) / (delta * delta));

/* 给定局数，能分辨的最小差距（80% power） */
const detectable = (p1, p2, R, deff = DEFF) =>
  Math.sqrt(((Z + ZP) ** 2 * diffVar(p1, p2) * deff) / R);

const pct = (x) => (x * 100).toFixed(1) + '%';
const pad = (s, w, right = true) => {
  s = String(s);
  let len = 0;
  for (const ch of s) len += /[\u4e00-\u9fff]/.test(ch) ? 2 : 1;
  const sp = ' '.repeat(Math.max(0, w - len));
  return right ? sp + s : s + sp;
};
const row = (cells, widths) => cells.map((c, i) => pad(c, widths[i], i > 0)).join(' ');

console.log(`\n设计效应 DEFF = ${DEFF}（同一出生布局下的对局相关，有效样本 = 名义局数 / DEFF）`);
console.log('z(95%)=1.96, power=80%\n');

/* ---------- 1. 单策略胜率精度 ---------- */
const N = parseInt(arg('n', '8'), 10);
console.log(`■ 8 人混战：想把某个策略的胜率测到 ±h，需要多少局（N=${N}）`);
console.log(row(['策略真实胜率', '±3pp', '±2pp', '±1.5pp', '±1pp'], [14, 10, 10, 10, 10]));
console.log('─'.repeat(56));
for (const p of [1 / N, 0.15, 0.2, 0.3, 0.4]) {
  const label = Math.abs(p - 1 / N) < 1e-9 ? pct(p) + '（全场均分）' : pct(p);
  console.log(row([label, roundsForHalfWidth(p, 0.03), roundsForHalfWidth(p, 0.02), roundsForHalfWidth(p, 0.015), roundsForHalfWidth(p, 0.01)], [14, 10, 10, 10, 10]));
}

/* ---------- 2. 两两分辨能力 ---------- */
console.log('\n■ 8 人混战：要在 80% power 下判定「A 真的比 B 强 Δ」，需要多少局');
console.log(row(['对比', 'Δ=8pp', 'Δ=5pp', 'Δ=3pp', 'Δ=2pp'], [22, 10, 10, 10, 10]));
console.log('─'.repeat(64));
for (const [a, b] of [[0.30, 0.20], [0.25, 0.25], [0.35, 0.25], [0.12, 0.08]]) {
  console.log(row([`${pct(a)} vs ${pct(b)}`, roundsForDiff(a, b, 0.08), roundsForDiff(a, b, 0.05), roundsForDiff(a, b, 0.03), roundsForDiff(a, b, 0.02)], [22, 10, 10, 10, 10]));
}

/* ---------- 3. 单挑 ---------- */
console.log('\n■ 1v1 单挑：p≈0.5 时方差最大，反而更费局');
console.log(row(['局数', '可分辨差距(80%)', '该对胜率 95%CI'], [8, 20, 20]));
console.log('─'.repeat(50));
for (const R of [50, 100, 200, 400, 800]) {
  console.log(row([R, '±' + (detectable(0.5, 0.5, R) * 100).toFixed(1) + 'pp', '±' + (Z * Math.sqrt(0.25 * DEFF / R) * 100).toFixed(1) + 'pp'], [8, 20, 20]));
}

/* ---------- 4. 给定预算，各指标精度 ---------- */
console.log('\n■ 反查：跑 R 局，主榜的 95% CI 有多宽（p=0.25 的最坏情况附近）');
console.log(row(['局数', 'CI 半宽'], [8, 12]));
console.log('─'.repeat(24));
for (const R of [200, 400, 800, 1600, 2400, 4000, 6000, 10000]) {
  console.log(row([R, '±' + (Z * Math.sqrt(0.25 * 0.75 * DEFF / R) * 100).toFixed(2) + 'pp'], [8, 12]));
}

console.log('\n口径提醒：');
console.log('  · 上面是「单看一个策略」的精度。同时看 8 个策略、做 28 次两两比较时，');
console.log('    误判率会膨胀；报告里请用配对的 cluster bootstrap 做差值 CI，而不是比较两个 CI 是否重叠。');
console.log('  · 局数取 2N 的倍数，才能让「每个策略 × 每个席位 × 两张图」完整配平。\n');
