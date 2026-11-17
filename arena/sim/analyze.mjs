/* ===========================================================
   sim/analyze.mjs — 战报分析
   -----------------------------------------------------------
   用法：
     node arena/sim/analyze.mjs                                   # 读 output/sim/rounds.jsonl
     node arena/sim/analyze.mjs --in a.jsonl,b.jsonl --boot 4000
     node arena/sim/analyze.mjs --tag main8

   四件事（随机模拟里不做就会白跑）：
     1. 置信区间按**出生布局聚类**重采样 —— 同一布局下的 N 局不是独立样本；
     2. 胜率 / 击杀率 / 自杀率 / 存活时长，按人数分表；
     3. Plackett–Luce 拟合潜在强度 —— 用来预测**没模拟过的名单**；
     4. 出生点分桶（开口数、离中心距离）—— 位置到底值多少胜率。
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');

const arg = (n, d) => {
  const i = process.argv.indexOf('--' + n);
  return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : d;
};
const IN = arg('in', path.join('output', 'sim', 'rounds.jsonl')).split(',').map((s) => path.resolve(ROOT, s.trim()));
const ONLY_TAG = arg('tag', '');
const BOOT = parseInt(arg('boot', '2000'), 10);
const SEED = parseInt(arg('seed', '1234567'), 10);
const OUT = path.resolve(ROOT, arg('out', path.join('output', 'sim')));

/* ---------------- 读入 ---------------- */

let rounds = [];
for (const f of IN) {
  if (!fs.existsSync(f)) { console.error('找不到 ' + f); process.exit(1); }
  for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { rounds.push(JSON.parse(line)); } catch { /* 半行 */ }
  }
}
if (!rounds.length) { console.error('没有对局记录'); process.exit(1); }
if (ONLY_TAG) rounds = rounds.filter((r) => r.tag === ONLY_TAG);

const bad = rounds.filter((r) => r.error);
const good = rounds.filter((r) => !r.error);

/* ---------------- 工具 ---------------- */

function mulberry32(a) {
  a = (a >>> 0) || 1;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(SEED);

const pct = (x) => (x * 100).toFixed(1) + '%';
const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);

function pad(s, w, right = false) {
  s = String(s);
  let len = 0;
  for (const ch of s) len += /[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/.test(ch) ? 2 : 1;
  const sp = ' '.repeat(Math.max(0, w - len));
  return right ? sp + s : s + sp;
}
function table(head, rows, widths) {
  console.log(head.map((h, i) => pad(h, widths[i], i > 0)).join(' '));
  console.log('─'.repeat(widths.reduce((a, b) => a + b, 0) + widths.length - 1));
  for (const r of rows) console.log(r.map((v, i) => pad(v, widths[i], i > 0)).join(' '));
}

const groupBy = (list, keyFn) => {
  const m = new Map();
  for (const x of list) {
    const k = keyFn(x);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(x);
  }
  return m;
};

/* 聚类键：同一张图 + 同一套出生坐标 = 一个簇（各名单共用布局 → 同簇） */
const clusterKey = (r) => r.n + ':' + r.mapSeed + ':' + r.spawnSeed;

/* ---------------- 描述统计 ---------------- */

function describe(set, ids) {
  const rows = {};
  for (const id of ids) rows[id] = { id, rounds: 0, wins: 0, kills: 0, suicide: 0, died: 0, life: 0 };
  for (const r of set) {
    for (const id of r.seats || []) {
      const row = rows[id];
      if (!row) continue;
      const died = r.deathTime && r.deathTime[id] != null;
      row.rounds++;
      if (r.winner === id) row.wins++;
      row.kills += (r.kills && r.kills[id]) || 0;
      row.suicide += (r.suicides && r.suicides[id]) || 0;
      row.died += died ? 1 : 0;
      row.life += died ? r.deathTime[id] : (r.durationSec || 0);
    }
  }
  for (const id in rows) {
    const r = rows[id];
    r.winRate = r.rounds ? r.wins / r.rounds : 0;
    r.killPerRound = r.rounds ? r.kills / r.rounds : 0;
    r.killPerMin = r.life ? (r.kills / r.life) * 60 : 0;
    r.suicidePerRound = r.rounds ? r.suicide / r.rounds : 0;
    r.deathRate = r.rounds ? r.died / r.rounds : 0;
    r.avgLife = r.rounds ? r.life / r.rounds : 0;
  }
  return rows;
}

/* ---------------- 聚类 bootstrap ---------------- */

function bootWinRate(set, ids, B = BOOT) {
  const clusters = [...groupBy(set, clusterKey).values()];
  const C = clusters.length;
  if (C < 3) return { cis: null, clusters: C };
  const acc = {};
  for (const id of ids) acc[id] = new Float64Array(B);

  for (let b = 0; b < B; b++) {
    const wins = {}, tot = {};
    for (const id of ids) { wins[id] = 0; tot[id] = 0; }
    for (let c = 0; c < C; c++) {
      for (const r of clusters[(rnd() * C) | 0]) {
        for (const id of r.seats || []) {
          if (!(id in tot)) continue;
          tot[id]++;
          if (r.winner === id) wins[id]++;
        }
      }
    }
    for (const id of ids) acc[id][b] = tot[id] ? wins[id] / tot[id] : 0;
  }

  const cis = {};
  const deff = {};
  for (const id of ids) {
    const raw = Array.from(acc[id]);
    const a = raw.slice().sort((x, y) => x - y);
    cis[id] = { lo: a[Math.floor(B * 0.025)], hi: a[Math.floor(B * 0.975)] };
    /* 设计效应 = bootstrap 方差 / 二项方差。>1 说明同一布局下的对局确实相关，
       名义局数要按这个倍数打折。 */
    const mean = raw.reduce((x, y) => x + y, 0) / raw.length;
    const v = raw.reduce((x, y) => x + (y - mean) ** 2, 0) / (raw.length - 1);
    const row = set.filter((r) => (r.seats || []).includes(id));
    const p = row.filter((r) => r.winner === id).length / (row.length || 1);
    const binom = p * (1 - p) / (row.length || 1);
    deff[id] = binom > 0 ? v / binom : 1;
  }
  return { cis, clusters: C, deff };
}

/* ---------------- Plackett–Luce ---------------- */

function fitPL(set, ids, iters = 600) {
  const idx = new Map(ids.map((id, i) => [id, i]));
  const k = ids.length;
  const theta = new Float64Array(k).fill(1);
  const wins = new Float64Array(k);
  const games = [];
  for (const r of set) {
    if (!r.seats || r.seats.length < 2) continue;
    const part = r.seats.map((id) => idx.get(id)).filter((i) => i != null);
    if (part.length < 2) continue;
    const w = r.winner != null ? idx.get(r.winner) : undefined;
    games.push({ part, w });
    if (w != null) wins[w]++;
  }
  if (!games.length || wins.reduce((a, b) => a + b, 0) < 5) return null;

  for (let it = 0; it < iters; it++) {
    const T = new Float64Array(k);
    for (const g of games) {
      let S = 0;
      for (const i of g.part) S += theta[i];
      if (S <= 0) continue;
      for (const i of g.part) T[i] += 1 / S;
    }
    let maxDelta = 0;
    for (let i = 0; i < k; i++) {
      const next = T[i] > 0 ? (wins[i] + 0.5) / (T[i] + 0.5) : 1;
      maxDelta = Math.max(maxDelta, Math.abs(next - theta[i]) / Math.max(1e-9, theta[i]));
      theta[i] = next;
    }
    const mean = theta.reduce((a, b) => a + b, 0) / k;
    for (let i = 0; i < k; i++) theta[i] /= mean;
    if (maxDelta < 1e-8) break;
  }

  const strength = {};
  ids.forEach((id, i) => { strength[id] = theta[i]; });
  const predict = (roster) => {
    const S = roster.reduce((a, id) => a + (strength[id] || 0), 0);
    return Object.fromEntries(roster.map((id) => [id, S ? strength[id] / S : 0]));
  };
  return { strength, predict, games: games.length };
}

/* ---------------- 单挑矩阵 ---------------- */

function duelMatrix(set, ids) {
  const cell = {};
  for (const a of ids) { cell[a] = {}; for (const b of ids) cell[a][b] = { w: 0, n: 0, d: 0 }; }
  for (const r of set) {
    if (!r.seats || r.seats.length !== 2) continue;
    const [a, b] = r.seats;
    if (!cell[a] || !cell[b]) continue;
    cell[a][b].n++; cell[b][a].n++;
    if (r.winner === a) cell[a][b].w++;
    else if (r.winner === b) cell[b][a].w++;
    else { cell[a][b].d++; cell[b][a].d++; }
  }
  return cell;
}

/* ---------------- 出生点分桶 ---------------- */

function spawnBuckets(set) {
  const out = [];
  for (const [n, list] of groupBy(set, (r) => r.n)) {
    if (list.length < 60) continue;
    const cells = [];
    for (const r of list) {
      for (const sp of r.spawns || []) {
        const seat = r.seats[sp.seat];
        if (!seat) continue;
        cells.push({
          id: seat, open: sp.open, x: sp.x, y: sp.y,
          dist: Math.hypot(sp.x - 640, sp.y - 400),
          won: r.winner === seat ? 1 : 0,
          kills: (r.kills && r.kills[seat]) || 0,
        });
      }
    }
    if (!cells.length) continue;

    const add = (label, fn) => {
      for (const [k, arr] of groupBy(cells, fn)) {
        if (arr.length < 40) continue;
        out.push({
          n, label: label + '=' + k, samples: arr.length,
          win: arr.reduce((a, c) => a + c.won, 0) / arr.length,
          kill: arr.reduce((a, c) => a + c.kills, 0) / arr.length,
        });
      }
    };
    add('开口数', (c) => c.open);
    add('离中心', (c) => (c.dist < 180 ? '近<180' : c.dist < 320 ? '中180-320' : c.dist < 450 ? '远320-450' : '极远>450'));
    add('象限', (c) => (c.x < 640 ? (c.y < 400 ? '左上' : '左下') : (c.y < 400 ? '右上' : '右下')));
  }
  return out;
}

/* ---------------- 主流程 ---------------- */

console.log(`\n读取 ${good.length} 局（异常 ${bad.length} 局）· 聚类 bootstrap ${BOOT} 次\n`);

const summary = { meta: { rounds: good.length, bad: bad.length, boot: BOOT, files: IN }, tags: {} };
const globalCurve = {};

for (const [tag, set] of groupBy(good, (r) => r.tag)) {
  const ns = [...new Set(set.map((r) => r.n))].sort((a, b) => a - b);
  const ids = [...new Set(set.flatMap((r) => r.seats || []))];
  const blocks = groupBy(set, clusterKey).size;

  console.log(`\n■ ${tag}  ·  ${set.length} 局  ·  ${blocks} 个出生布局簇  ·  ${ns.join('/')} 人`);
  summary.tags[tag] = { rounds: set.length, clusters: blocks, ns, byN: {} };

  /* 单挑矩阵 */
  if (tag === 'duel' && ids.length >= 2) {
    const cell = duelMatrix(set, ids);
    const order = ids.slice().sort((a, b) => {
      const wa = ids.reduce((s, x) => s + (cell[a][x].n ? cell[a][x].w / cell[a][x].n : 0), 0);
      const wb = ids.reduce((s, x) => s + (cell[b][x].n ? cell[b][x].w / cell[b][x].n : 0), 0);
      return wb - wa;
    });
    const head = ['胜\\负', ...order.map((x) => x.slice(0, 5))];
    table(head, order.map((a) => [a.slice(0, 10), ...order.map((b) => (a === b ? '—' : cell[a][b].n ? pct(cell[a][b].w / cell[a][b].n) : '·'))]),
      [11, ...order.map(() => 7)]);
    console.log('   每格 = 行方胜率（对角线为自身；平局计入分母）');
  }

  /* 按人数分表 */
  for (const n of ns) {
    const sub = set.filter((r) => r.n === n);
    /* 只统计在该人数下真的出过场的策略：没出场的策略若留在 PL 里，
       强度会停在 1.00 白占分母，把所有人的预测胜率都压低。 */
    const idsN = [...new Set(sub.flatMap((r) => r.seats || []))];
    const row = describe(sub, idsN);
    const { cis, clusters, deff } = bootWinRate(sub, idsN);
    const list = Object.values(row).filter((r) => r.rounds > 0).sort((a, b) => b.winRate - a.winRate);

    if (ns.length > 1) console.log(`\n  — ${n} 人局（${sub.length} 局 / ${clusters} 簇）`);
    table(
      ['策略', '局数', '胜率', '95% CI', '击杀/局', '击杀/分', '自杀/局', '阵亡率', '存活'],
      list.map((r) => [
        r.id, r.rounds, pct(r.winRate),
        cis ? `${pct(cis[r.id].lo)}~${pct(cis[r.id].hi)}` : '—',
        f3(r.killPerRound), f3(r.killPerMin), f3(r.suicidePerRound),
        pct(r.deathRate), r.avgLife.toFixed(1) + 's',
      ]),
      [11, 6, 7, 15, 8, 8, 8, 7, 7]
    );
    if (deff) {
      const ds = Object.values(deff).filter((x) => Number.isFinite(x) && x > 0).sort((a, b) => a - b);
      if (ds.length) console.log(`   设计效应 DEFF ≈ ${f2(ds[Math.floor(ds.length / 2)])}（中位数；有效样本 ≈ ${Math.round(sub.length / Math.max(1, ds[Math.floor(ds.length / 2)]))} 局）`);
    }

    const pl = fitPL(sub, idsN);
    if (pl) {
      const rank = idsN.slice().sort((a, b) => pl.strength[b] - pl.strength[a]);
      console.log(`\n   PL 潜在强度：` + rank.map((id) => `${id} ${f2(pl.strength[id])}`).join(' · '));

      /* 校准：按名单把预测与实际对照 */
      let se = 0, cnt = 0;
      const cal = [];
      for (const [key, rs] of groupBy(sub, (r) => (r.seats || []).slice().sort().join('+'))) {
        if (rs.length < 20) continue;
        const pred = pl.predict(key.split('+'));
        for (const id of key.split('+')) {
          const obs = rs.filter((r) => r.winner === id).length / rs.length;
          se += (obs - pred[id]) ** 2; cnt++;
          cal.push({ roster: key, id, obs, pred: pred[id] });
        }
      }
      if (cnt) console.log(`   校准：${cnt} 个(名单,策略)格 · RMSE ${(Math.sqrt(se / cnt) * 100).toFixed(2)}pp`);

      summary.tags[tag].byN[n] = {
        rounds: sub.length, clusters,
        rows: Object.fromEntries(list.map((r) => [r.id, { ...r, ciLo: cis ? cis[r.id].lo : null, ciHi: cis ? cis[r.id].hi : null }])),
        strength: pl.strength, calibration: cal,
      };
    }
  }
}

/* 人数曲线：直接看「同一策略在不同人数下的实测胜率」 */
const curveNs = [...new Set(good.map((r) => r.n))].sort((a, b) => a - b);
if (curveNs.length > 1) {
  const curve = {};
  for (const n of curveNs) {
    curve[n] = {};
    for (const r of good.filter((x) => x.n === n)) {
      for (const id of r.seats || []) {
        const c = (curve[n][id] ||= { w: 0, n: 0 });
        c.n++; if (r.winner === id) c.w++;
      }
    }
  }
  const order = Object.keys(curve[curveNs[curveNs.length - 1]] || {})
    .sort((a, b) => (curve[curveNs[curveNs.length - 1]][b].w / curve[curveNs[curveNs.length - 1]][b].n) -
                    (curve[curveNs[curveNs.length - 1]][a].w / curve[curveNs[curveNs.length - 1]][a].n));

  console.log('\n■ 人数曲线：实测胜率（每格 = 该策略在该人数下的胜率 / 样本局数）');
  table(
    ['策略', ...curveNs.map((n) => n + ' 人')],
    order.map((id) => [id, ...curveNs.map((n) => {
      const c = curve[n][id];
      return c ? pct(c.w / c.n) + '/' + c.n : '—';
    })]),
    [11, ...curveNs.map(() => 10)]
  );
  console.log('   基线 = 1/N（' + curveNs.map((n) => n + '人 ' + (100 / n).toFixed(1) + '%').join(' · ') + '）');
  console.log('   注意：不同 N 下的名单不同，胜率差异里混着「队友/对手变了」的效应，不是纯人数效应；');
  console.log('   要分离人数效应，看 count 阶段里同一策略在不同 N 的 PL 强度变化。');
  summary.curve = { ns: curveNs, values: curve };
}

/* 全局 PL：用来预测没跑过的名单 */
const allIds = [...new Set(good.flatMap((r) => r.seats || []))];
const gpl = fitPL(good, allIds);
if (gpl) {
  summary.globalStrength = gpl.strength;
  console.log('\n■ 全局 Plackett–Luce 强度（所有人数合并拟合，1.00 = 全场平均）');
  console.log('   ' + allIds.slice().sort((a, b) => gpl.strength[b] - gpl.strength[a])
    .map((id) => `${id} ${f2(gpl.strength[id])}`).join('  ·  '));

  const want = arg('predict', '');
  if (want) {
    const roster = want.split(',').map((x) => x.trim()).filter((x) => gpl.strength[x] != null);
    if (roster.length >= 2) {
      console.log(`\n   预测名单 [${roster.join(', ')}] 的夺冠概率：`);
      const p = gpl.predict(roster);
      table(['策略', '预测胜率'], roster.slice().sort((a, b) => p[b] - p[a]).map((id) => [id, pct(p[id])]), [11, 9]);
    } else console.log('   --predict 需要 2 个以上已知策略');
  } else {
    console.log('   想预测某个未模拟的名单？加 --predict gpt,hy,qwen');
  }
}

/* 出生点 */
const sb = spawnBuckets(good);
if (sb.length) {
  console.log('\n■ 出生点效应（同人数内比较；开口数 = 该格朝外通路，3~4 开阔，2 半死角）');
  table(
    ['人数', '分桶', '样本', '胜率', '击杀/局'],
    sb.map((s) => [s.n, s.label, s.samples, pct(s.win), f3(s.kill)]),
    [6, 14, 7, 7, 8]
  );
  console.log('   注：先看哪个维度真的分得开。pilot 里「开口数 3 vs 4」几乎无差异，');
  console.log('       而「离中心距离」梯度明显（越靠中心越吃亏）——这种维度才值得继续加样本。');
  summary.spawn = sb;
}

/* 落盘 */
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));

const csv = [['tag', 'n', 'policy', 'rounds', 'wins', 'winRate', 'ciLo', 'ciHi', 'killsPerRound', 'killsPerMin', 'suicidePerRound', 'deathRate', 'avgLife'].join(',')];
for (const [tag, s] of Object.entries(summary.tags)) {
  for (const [n, d] of Object.entries(s.byN)) {
    for (const r of Object.values(d.rows)) {
      csv.push([tag, n, r.id, r.rounds, r.wins, r.winRate.toFixed(4),
        r.ciLo == null ? '' : r.ciLo.toFixed(4), r.ciHi == null ? '' : r.ciHi.toFixed(4),
        r.killPerRound.toFixed(3), r.killPerMin.toFixed(3), r.suicidePerRound.toFixed(3), r.deathRate.toFixed(3), r.avgLife.toFixed(2)].join(','));
    }
  }
}
fs.writeFileSync(path.join(OUT, 'summary.csv'), csv.join('\n'));
console.log(`\n已写入 ${path.relative(ROOT, path.join(OUT, 'summary.json'))} 与 summary.csv\n`);
