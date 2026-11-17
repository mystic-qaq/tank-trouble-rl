/* ===========================================================
   sim/plan.mjs — 实验矩阵生成
   -----------------------------------------------------------
   把「想知道什么」翻译成一张 (人数, 名单, 种子, 局数) 的任务表。

   三层设计：
     main8  8 人全阵容主榜                  —— 回答「谁强」
     count  N=3..7 人数梯度                 —— 回答「人多了谁受益」
     duel   28 对单挑矩阵                   —— 回答「两两谁克谁」

   平衡性：
     · 每个任务局数都是 N 的倍数 → 每个策略在每个席位恰好出现相同次数；
       是 2N 的倍数时两张地图也配平。
     · count 阶段同一 N 的所有名单共用同一个 baseSeed，因此**出生布局逐局对齐**，
       名单之间是可配对比较（common random numbers），方差大幅下降。
     · duel 阶段 28 对共用同一个 baseSeed，所有对局看到完全相同的出生布局序列。
   =========================================================== */

export const ALL_IDS = ['deepseek', 'glm', 'gpt', 'hy', 'kimi', 'minimax', 'qwen', 'builtin'];

/* 单核实测墙钟（秒/局，i9-13980HX 上实测）。只用于 ETA，不参与统计。 */
export const COST = { 2: 1.5, 3: 2.8, 4: 4.5, 5: 4.5, 6: 4.5, 7: 6.0, 8: 6.0 };

/* 实测并行效率：这台机器上 8~24 并发都只跑到 ~5.5 个「单核速度」的聚合吞吐，
   瓶颈是缓存/内存带宽，不是核数。ETA 用它而不是 JOBS。 */
export const EFFECTIVE_CORES = 5.5;

/* 目标局数的四档预算 */
export const BUDGETS = {
  smoke:    { main8: 32,   count: 1,  duel: 3,   minBlocks: 2, note: '自检：验证流水线通不通，不看结论' },
  pilot:    { main8: 320,  count: 4,  duel: 8,   minBlocks: 6, note: '摸底：误差 ±5~8pp，看趋势' },
  quick:    { main8: 800,  count: 12, duel: 40,  minBlocks: 6, note: '粗排：分得清强/中/弱，误差 ±3~4pp' },
  standard: { main8: 2400, count: 18, duel: 80,  minBlocks: 6, note: '可发布：主榜 ±2pp，能分辨 5pp 差距' },
  deep:     { main8: 6000, count: 30, duel: 200, minBlocks: 10, note: '卡位战：主榜 ±1.2pp，能分辨 3pp 差距' },
};

/* 最小公倍数式的向上取整：保证局数同时是 N 和 2N 的倍数 */
function align(rounds, n) {
  const unit = 2 * n;
  return Math.max(unit, Math.round(rounds / unit) * unit);
}

/**
 * 贪心平衡抽名单：每次挑「当前出场次数最少」的 n 个策略，再局部修补到完全均衡。
 * 结果：每个策略出场次数差 ≤ 1；对 n=2 会自然生成近似循环赛。
 */
export function balancedRosters(n, count, ids = ALL_IDS) {
  if (n === ids.length) return [ids.slice()];
  const appear = Object.fromEntries(ids.map((id) => [id, 0]));
  const seen = new Set();
  const out = [];
  let guard = 0;
  while (out.length < count && guard++ < count * 30) {
    const order = ids.slice().sort((a, b) => {
      const d = appear[a] - appear[b];
      return d !== 0 ? d : (a < b ? -1 : 1);
    });
    let pick = order.slice(0, n).sort((a, b) => ids.indexOf(a) - ids.indexOf(b));
    if (seen.has(pick.join(','))) {
      let swapped = false;
      for (let k = 0; k < n && !swapped; k++) {
        for (const alt of order.slice(n)) {
          let cand = pick.slice();
          cand[k] = alt;
          cand = cand.sort((a, b) => ids.indexOf(a) - ids.indexOf(b));
          if (!seen.has(cand.join(','))) { pick = cand; swapped = true; break; }
        }
      }
      if (!swapped) break;
    }
    seen.add(pick.join(','));
    out.push(pick);
    for (const id of pick) appear[id]++;
  }

  /* 修补：把出场最多的策略换进「出场最少且不在名单里」的位置，直到极差 ≤ 1。
     贪心 + 去重的过程中难免抖出 ±3 的不平衡，这里抹平，避免边际胜率被名单结构污染。 */
  for (let pass = 0; pass < 2000; pass++) {
    const maxId = ids.reduce((a, b) => (appear[a] >= appear[b] ? a : b));
    const minId = ids.reduce((a, b) => (appear[a] <= appear[b] ? a : b));
    if (appear[maxId] - appear[minId] <= 1) break;
    let fixed = false;
    for (const roster of out) {
      if (!roster.includes(maxId) || roster.includes(minId)) continue;
      const cand = roster.map((id) => (id === maxId ? minId : id))
        .sort((a, b) => ids.indexOf(a) - ids.indexOf(b));
      const key = cand.join(',');
      if (seen.has(key) && key !== roster.join(',')) continue;
      seen.delete(roster.join(','));
      seen.add(key);
      roster.splice(0, roster.length, ...cand);
      appear[maxId]--; appear[minId]++;
      fixed = true;
      break;
    }
    if (!fixed) break;
  }
  return out;
}

/** 28 个两两组合，保持稳定顺序 */
export function allPairs(ids = ALL_IDS) {
  const out = [];
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) out.push([ids[i], ids[j]]);
  }
  return out;
}

/**
 * @param {object} opt
 * @param {string} opt.budget  BUDGETS 的键
 * @param {number} opt.seed    基础种子（所有阶段共用，制造配对结构）
 * @param {string[]} opt.ids   参赛池
 * @param {string[]} opt.stages 要跑的阶段，缺省全跑
 */
export function buildPlan({ budget = 'standard', seed = 20260101, ids = ALL_IDS, stages = ['main8', 'count', 'duel'] } = {}) {
  const cfg = BUDGETS[budget];
  if (!cfg) throw new Error('未知预算档：' + budget + '（可选 ' + Object.keys(BUDGETS).join('/') + '）');
  if (ids.length < 2) throw new Error('参赛池至少 2 个策略');

  const jobs = [];
  const push = (tag, players, rounds, extra = {}) => {
    jobs.push({
      id: `${tag}|${players.join('+')}`,
      tag, players: players.slice(), rounds,
      seed, ...extra,
    });
  };

  if (stages.includes('main8')) {
    if (ids.length >= 2) {
      const n = ids.length;
      push('main8', ids, align(cfg.main8, n));
    }
  }

  if (stages.includes('count')) {
    for (let n = 3; n <= Math.min(7, ids.length - 1); n++) {
      const rosters = balancedRosters(n, cfg.count, ids);
      /* 每个名单至少要有 6 个不同出生布局，bootstrap 才有意义：
         「每 N 局一个布局」意味着一个布局簇里只有 N 局，簇太少时 CI 直接失效。 */
      const per = align(Math.max(Math.ceil(cfg.main8 / 4 / rosters.length), (cfg.minBlocks || 6) * n), n);
      for (const r of rosters) push('count', r, per, { n });
    }
  }

  if (stages.includes('duel')) {
    for (const pair of allPairs(ids)) push('duel', pair, align(cfg.duel, 2));
  }

  const cost = jobs.reduce((a, j) => a + j.rounds * (COST[j.players.length] || 1.2), 0);
  return { budget, seed, ids, cfg, jobs, cpuSeconds: cost };
}

export function planSummary(plan) {
  const byTag = {};
  for (const j of plan.jobs) {
    const t = (byTag[j.tag] ||= { jobs: 0, rounds: 0, seconds: 0 });
    t.jobs++; t.rounds += j.rounds; t.seconds += j.rounds * (COST[j.players.length] || 1.2);
  }
  return byTag;
}
