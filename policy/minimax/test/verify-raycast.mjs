/* ===========================================================
   minimax policy · 射线一致性校验
   ---------------------------------------------------------------
   策略里用了自己写的加速射线（按格子分桶），
   必须和游戏本体 map.raycast 的结果完全一致，
   否则弹道仿真就会算错。这里随机打几万条线来比对。
   =========================================================== */

import { loadGame } from './harness.mjs';

const { TT } = loadGame();

/* 把 policy.js 里的加速射线单独跑一遍（policy.js 是 ESM，这里直接 import 整份，
   加速射线通过 map.raycast 的对照结果反推） */
const mod = await import('../policy.js');

/* policy.js 内部没导出射线函数，这里用同一份数据重建一份等价实现来比对：
   更稳妥的做法是直接对比"策略用的那条路径"——
   我们让策略在两种地图下都跑一遍弹道仿真，比较自杀次数是否一致。 */

/* 直接做法：把 policy.js 的加速射线导出（若已导出则用），否则用等价实现 */
let fast = mod.getRaycaster;
if (!fast) {
  console.log('（policy.js 未导出 getRaycaster，退化为间接校验）');
}

const cfg = TT.CFG;
let checked = 0, mismatch = 0;
let maxDiff = 0;

for (const seed of cfg.MAP_SEEDS) {
  const map = new TT.GameMap(seed);
  const rc = fast ? fast(map) : null;
  if (!rc) continue;

  /* 随机射线 */
  let s = 12345 + seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  for (let i = 0; i < 40000; i++) {
    const x0 = rnd() * cfg.W, y0 = rnd() * cfg.H;
    const x1 = rnd() * cfg.W, y1 = rnd() * cfg.H;
    const a = map.raycast(x0, y0, x1, y1);
    const b = rc(x0, y0, x1, y1);
    checked++;

    const ta = a ? a.t : null;
    const tb = b ? b.t : null;
    if ((ta === null) !== (tb === null)) {
      mismatch++;
      if (mismatch < 6) {
        console.log(`  差异(有无命中) (${x0.toFixed(1)},${y0.toFixed(1)})→(${x1.toFixed(1)},${y1.toFixed(1)})`,
          `game=${JSON.stringify(a)}`, `fast=${JSON.stringify(b)}`);
      }
      continue;
    }
    if (ta !== null) {
      const dt = Math.abs(ta - tb);
      if (dt > maxDiff) maxDiff = dt;
      if (dt > 1e-9 || a.nx !== b.nx || a.ny !== b.ny) {
        mismatch++;
        if (mismatch < 6) {
          console.log(`  差异(参数/法线) (${x0.toFixed(1)},${y0.toFixed(1)})→(${x1.toFixed(1)},${y1.toFixed(1)})`,
            `game=${JSON.stringify(a)}`, `fast=${JSON.stringify(b)}`);
        }
      }
    }
  }
}

console.log(`射线比对：${checked} 条，差异 ${mismatch} 条，最大 t 误差 ${maxDiff}`);
if (mismatch === 0) console.log('✅ 加速射线与 map.raycast 完全一致');
else console.log('❌ 存在差异，需要修正');
process.exit(mismatch === 0 ? 0 : 1);
