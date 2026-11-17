/* navtest.mjs — 导航表单元自检：next[] 必须真的沿空格子走到终点 */
import path from 'node:path';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadTT } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');

const { ctx, TT, canvas } = loadTT({ seed: 4242 });
vm.runInContext(fs.readFileSync(POLICY, 'utf8'), ctx, { filename: POLICY });
const Q = TT.policies.qwen;

let checked = 0, bad = 0, worstLen = 0;
const g = new TT.Game(canvas);
g.render = () => {}; g.resize = () => {};

for (const seed of TT.CFG.MAP_SEEDS) {
  const map = new TT.GameMap(seed);
  const model = Q._internals.models.get(seed + ':' + map.cols + 'x' + map.rows + ':' + map.walls.length)
    || new Q._internals.MapModel(map);
  const n = model.n, cols = model.cols;
  for (let s = 0; s < n; s++) {
    for (let t = 0; t < n; t++) {
      const d = model.steps[s * n + t];
      if (d < 0) continue;
      if (d === 0) continue;
      checked++;
      let cur = s, len = 0, ok = true;
      while (cur !== t && len < d + 4) {
        const nx = model.next[cur * n + t];
        if (nx < 0 || nx === cur) { ok = false; break; }
        /* 必须是相邻且没墙 */
        const cr = (cur / cols) | 0, cc = cur % cols;
        const nr = (nx / cols) | 0, nc = nx % cols;
        const adj = Math.abs(cr - nr) + Math.abs(cc - nc) === 1;
        if (!adj) { ok = false; break; }
        const open = model.open[cur];
        const blocked = (nc === cc + 1 && !(open & 8)) || (nc === cc - 1 && !(open & 4)) ||
                        (nr === cr + 1 && !(open & 2)) || (nr === cr - 1 && !(open & 1));
        if (blocked) { ok = false; break; }
        cur = nx; len++;
      }
      if (!ok || cur !== t) { bad++; if (bad < 4) console.log('  坏路径 seed', seed, 's', s, 't', t, 'd', d, '走了', len); }
      if (len > worstLen) worstLen = len;
    }
  }
}
console.log(`导航自检: 检查 ${checked} 对格子，坏路径 ${bad} 条，最长实际步数 ${worstLen}`);
process.exit(bad === 0 ? 0 : 1);
