/* ===========================================================
   minimax policy · 出生点统计
   看看第 k 个被挑中的出生格（= P(k+1)）在地图上是不是系统性更好。
   =========================================================== */

import { loadGame } from './harness.mjs';

const { TT } = loadGame();
const cfg = TT.CFG;
const N = 4000;

function visCount(map, r, c) {
  const px = c * cfg.CELL + cfg.CELL / 2;
  const py = r * cfg.CELL + cfg.CELL / 2;
  let n = 0;
  for (let rr = 1; rr < cfg.ROWS - 1; rr++) {
    for (let cc = 1; cc < cfg.COLS - 1; cc++) {
      const qx = cc * cfg.CELL + cfg.CELL / 2;
      const qy = rr * cfg.CELL + cfg.CELL / 2;
      if (map.hasClearShot(px, py, qx, qy, 16)) n++;
    }
  }
  return n;
}

const stats = [];
for (let i = 0; i < 8; i++) stats.push({ open: 0, vis: 0, dist: 0, n: 0 });

for (const seed of cfg.MAP_SEEDS) {
  const map = new TT.GameMap(seed);
  for (let t = 0; t < N; t++) {
    const spawns = map.pickSpawns(8, Math.random);
    if (spawns.length < 8) continue;
    for (let i = 0; i < 8; i++) {
      const s = spawns[i];
      const c = Math.min(cfg.COLS - 2, Math.max(1, Math.floor(s.x / cfg.CELL)));
      const r = Math.min(cfg.ROWS - 2, Math.max(1, Math.floor(s.y / cfg.CELL)));
      const st = stats[i];
      st.open += map.openSides(r, c);
      st.vis += visCount(map, r, c);
      st.dist += Math.hypot(s.x - cfg.W / 2, s.y - cfg.H / 2);
      st.n++;
    }
  }
}

console.log('座位  平均开放边  平均可见格数  距地图中心距离');
console.log('─'.repeat(52));
for (let i = 0; i < 8; i++) {
  const s = stats[i];
  console.log(
    ` P${i + 1}      ${(s.open / s.n).toFixed(2).padStart(6)}` +
    `      ${(s.vis / s.n).toFixed(1).padStart(8)}` +
    `        ${(s.dist / s.n).toFixed(0).padStart(6)}`
  );
}
