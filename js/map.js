/* ===========================================================
   map.js — 原版 TT 风格地图：落在网格边线上的细墙
   -----------------------------------------------------------
   单元格 80×80，墙体 10px 厚，全部贴在单元格之间的边界上，
   连续段合并成一条长墙。地图由递归分割生成，完全不对称，
   再挖出若干房间、打通少量墙形成回路。
   =========================================================== */

(function () {
  const CFG = TT.CFG;
  const U = TT.util;
  const P = TT.PALETTE;

  const COLS = CFG.COLS;      // 12
  const ROWS = CFG.ROWS;      // 9
  const CELL = CFG.CELL;      // 80
  const T = CFG.WALL_THICK;   // 16

  const VC = COLS + 1;        // 竖墙边界数量
  const HC = ROWS + 1;        // 横墙边界数量

  TT.GameMap = class GameMap {
    constructor(seed) {
      this.seed = (seed >>> 0) || 1;
      this.cols = COLS;
      this.rows = ROWS;
      this.cell = CELL;
      this.w = CFG.W;
      this.h = CFG.H;
      this.thick = T;

      this.vFlag = new Uint8Array(ROWS * VC);
      this.hFlag = new Uint8Array(HC * COLS);
      this.walls = [];

      this.generate();
    }

    vAt(r, c) { return this.vFlag[r * VC + c]; }
    hAt(r, c) { return this.hFlag[r * COLS + c]; }

    /* ---------------- 生成 ---------------- */

    generate() {
      let bestV = null, bestH = null, best = -1;

      for (let attempt = 0; attempt < 70; attempt++) {
        const rnd = U.rng(this.seed + attempt * 7919);
        this.buildFlags(rnd);
        const score = this.score();
        if (score > best) {
          best = score;
          bestV = this.vFlag.slice();
          bestH = this.hFlag.slice();
        }
        if (score >= 0.94) break;
      }

      if (best >= 0 && best < 0.94) {
        this.vFlag = bestV;
        this.hFlag = bestH;
      }

      this.buildWalls();
    }

    /* 递归分割：长墙 + 至少一个缺口，天然连通且完全不对称 */
    buildFlags(rnd) {
      this.vFlag.fill(0);
      this.hFlag.fill(0);

      const divide = (r0, c0, r1, c1, depth) => {
        const h = r1 - r0;
        const w = c1 - c0;
        if (h < 2 && w < 2) return;
        if (depth > 18) return;

        let vertical;
        if (w < 2) vertical = false;
        else if (h < 2) vertical = true;
        else if (w > h * 1.3) vertical = true;
        else if (h > w * 1.3) vertical = false;
        else vertical = rnd() < 0.42;   /* 略偏横向，长墙更接近原版 */

        if (vertical) {
          const cs = c0 + 1 + Math.floor((w - 1) * (0.3 + rnd() * 0.4));
          for (let r = r0; r < r1; r++) this.vFlag[r * VC + cs] = 1;

          /* 一条长墙上留 1~2 个口子 */
          const gaps = (r1 - r0 >= 5 && rnd() < 0.28) ? 2 : 1;
          for (let g = 0; g < gaps; g++) {
            const rr = r0 + Math.floor(rnd() * h);
            this.vFlag[rr * VC + cs] = 0;
          }
          divide(r0, c0, r1, cs, depth + 1);
          divide(r0, cs, r1, c1, depth + 1);
        } else {
          const rs = r0 + 1 + Math.floor((h - 1) * (0.3 + rnd() * 0.4));
          for (let c = c0; c < c1; c++) this.hFlag[rs * COLS + c] = 1;

          const gaps = (c1 - c0 >= 5 && rnd() < 0.28) ? 2 : 1;
          for (let g = 0; g < gaps; g++) {
            const cc = c0 + Math.floor(rnd() * w);
            this.hFlag[rs * COLS + cc] = 0;
          }
          divide(r0, c0, rs, c1, depth + 1);
          divide(rs, c0, r1, c1, depth + 1);
        }
      };

      divide(0, 0, ROWS, COLS, 0);

      /* 挖出若干开阔房间：矩形内部的墙全部清掉 */
      const rooms = 6 + Math.floor(rnd() * 4);
      for (let i = 0; i < rooms; i++) {
        const rw = 2 + (rnd() < 0.34 ? 1 : 0);
        const rh = 2 + (rnd() < 0.28 ? 1 : 0);
        const r0 = 1 + Math.floor(rnd() * (ROWS - 1 - rh));
        const c0 = 1 + Math.floor(rnd() * (COLS - 1 - rw));

        for (let r = r0; r < r0 + rh; r++) {
          for (let c = c0 + 1; c < c0 + rw; c++) this.vFlag[r * VC + c] = 0;
        }
        for (let r = r0 + 1; r < r0 + rh; r++) {
          for (let c = c0; c < c0 + rw; c++) this.hFlag[r * COLS + c] = 0;
        }
      }

      /* 再随机拆掉少量墙，制造回路与横向通道 */
      for (let r = 0; r < ROWS; r++) {
        for (let c = 1; c < COLS; c++) {
          if (this.vFlag[r * VC + c] && rnd() < 0.10) this.vFlag[r * VC + c] = 0;
        }
      }
      for (let r = 1; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (this.hFlag[r * COLS + c] && rnd() < 0.10) this.hFlag[r * COLS + c] = 0;
        }
      }

      /* 少量独立的短墙，让结构更碎、更有原版那种参差感 */
      const stubs = 3 + Math.floor(rnd() * 4);
      for (let i = 0; i < stubs; i++) {
        const r = 1 + Math.floor(rnd() * (ROWS - 2));
        const c = 1 + Math.floor(rnd() * (COLS - 2));
        const v = rnd() < 0.5;
        const slot = v ? (r * VC + c) : (r * COLS + c);
        const arr = v ? this.vFlag : this.hFlag;
        if (arr[slot]) continue;
        if (v && (this.vAt(r - 1, c) || this.vAt(r + 1, c))) continue;
        if (!v && (this.hAt(r, c - 1) || this.hAt(r, c + 1))) continue;

        arr[slot] = 1;
        if (this.reachRatio() < 1) arr[slot] = 0;   /* 不许把任何格子封死 */
      }
    }

    reachRatio() {
      const total = ROWS * COLS;
      const seen = new Uint8Array(total);
      const stack = [1 * COLS + 1];
      seen[stack[0]] = 1;
      let count = 0;

      while (stack.length) {
        const idx = stack.pop();
        count++;
        const r = (idx / COLS) | 0;
        const c = idx % COLS;

        if (c < COLS - 1 && !this.vAt(r, c + 1) && !seen[idx + 1]) {
          seen[idx + 1] = 1; stack.push(idx + 1);
        }
        if (c > 0 && !this.vAt(r, c) && !seen[idx - 1]) {
          seen[idx - 1] = 1; stack.push(idx - 1);
        }
        if (r < ROWS - 1 && !this.hAt(r + 1, c) && !seen[idx + COLS]) {
          seen[idx + COLS] = 1; stack.push(idx + COLS);
        }
        if (r > 0 && !this.hAt(r, c) && !seen[idx - COLS]) {
          seen[idx - COLS] = 1; stack.push(idx - COLS);
        }
      }

      return count / total;
    }

    openSides(r, c) {
      let n = 0;
      if (!this.hAt(r, c)) n++;
      if (!this.hAt(r + 1, c)) n++;
      if (!this.vAt(r, c)) n++;
      if (!this.vAt(r, c + 1)) n++;
      return n;
    }

    /* 两点之间是否有一条能直射的通道（含车体宽度余量） */
    hasClearShot(x0, y0, x1, y1, pad) {
      const dx = x1 - x0, dy = y1 - y0;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const nx = -dy / len, ny = dx / len;
      const p = pad == null ? 16 : pad;
      const offs = [0, p, -p];

      for (let i = 0; i < offs.length; i++) {
        const o = offs[i];
        if (!this.raycast(x0 + nx * o, y0 + ny * o, x1 + nx * o, y1 + ny * o)) return true;
      }
      return false;
    }

    /* 每回合随机挑选出生格：互相拉开距离，且两两之间不能直瞄 */
    pickSpawns(n, rnd) {
      const rand = rnd || Math.random;
      const cands = [];
      for (let r = 1; r < ROWS - 1; r++) {
        for (let c = 1; c < COLS - 1; c++) {
          if (this.openSides(r, c) >= 3) cands.push({ r: r, c: c });
        }
      }
      if (cands.length < n * 5) {
        for (let r = 1; r < ROWS - 1; r++) {
          for (let c = 1; c < COLS - 1; c++) {
            if (this.openSides(r, c) === 2) cands.push({ r: r, c: c });
          }
        }
      }

      const px = function (p) { return p.c * CELL + CELL / 2; };
      const py = function (p) { return p.r * CELL + CELL / 2; };

      let best = [];

      for (let attempt = 0; attempt < 24; attempt++) {
        for (let i = cands.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          const t = cands[i]; cands[i] = cands[j]; cands[j] = t;
        }

        const chosen = [];
        let minD = 9;

        while (chosen.length < n && minD >= 2) {
          for (let i = 0; i < cands.length && chosen.length < n; i++) {
            const p = cands[i];
            let ok = true;

            for (let k = 0; k < chosen.length; k++) {
              const q = chosen[k];
              if (Math.abs(p.r - q.r) + Math.abs(p.c - q.c) < minD) { ok = false; break; }
              if (this.hasClearShot(px(p), py(p), px(q), py(q), 16)) { ok = false; break; }
            }
            if (ok) chosen.push(p);
          }
          if (chosen.length < n) minD--;   /* 只放宽距离，绝不放宽视线 */
        }

        if (chosen.length === n) { best = chosen; break; }
        if (chosen.length > best.length) best = chosen;
      }

      /* 极端情况下的兜底：补齐到 n，但尽量选看不见的位置 */
      if (best.length < n) {
        for (let i = 0; i < cands.length && best.length < n; i++) {
          const p = cands[i];
          let dup = false;
          for (let k = 0; k < best.length; k++) {
            if (best[k].r === p.r && best[k].c === p.c) { dup = true; break; }
          }
          if (dup) continue;
          let blocked = true;
          for (let k = 0; k < best.length; k++) {
            const q = best[k];
            if (this.hasClearShot(px(p), py(p), px(q), py(q), 16)) { blocked = false; break; }
          }
          if (blocked) best.push(p);
        }
        for (let i = 0; i < cands.length && best.length < n; i++) {
          const p = cands[i];
          let dup = false;
          for (let k = 0; k < best.length; k++) {
            if (best[k].r === p.r && best[k].c === p.c) { dup = true; break; }
          }
          if (!dup) best.push(p);
        }
      }

      const cx = this.w / 2, cy = this.h / 2;
      const out = [];
      for (let i = 0; i < best.length && i < n; i++) {
        const p = best[i];
        const x = px(p), y = py(p);

        /* 出生朝向：优先选一面对着地图中心、且真的通的口 */
        const toCenter = Math.atan2(cy - y, cx - x);
        const dirs = [
          { a: 0, open: !this.vAt(p.r, p.c + 1) },
          { a: Math.PI / 2, open: !this.hAt(p.r + 1, p.c) },
          { a: Math.PI, open: !this.vAt(p.r, p.c) },
          { a: -Math.PI / 2, open: !this.hAt(p.r, p.c) },
        ];
        let pickA = toCenter;
        let bestDiff = Infinity;
        for (let k = 0; k < dirs.length; k++) {
          if (!dirs[k].open) continue;
          const d = Math.abs(U.angleDiff(toCenter, dirs[k].a));
          if (d < bestDiff) { bestDiff = d; pickA = dirs[k].a; }
        }

        out.push({
          x: x, y: y,
          a: pickA + (rand() - 0.5) * 0.42,
        });
      }
      /* 贪心挑选顺序不代表玩家席位：随机分配整组出生点，消除先选优势。 */
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    }

    /* 地图评分：连通性 + 墙体密度 + 可出生点数量 */
    score() {
      const reach = this.reachRatio();
      if (reach < 1) return reach * 0.55;

      let walls = 0;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 1; c < COLS; c++) if (this.vFlag[r * VC + c]) walls++;
      }
      for (let r = 1; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) if (this.hFlag[r * COLS + c]) walls++;
      }
      const density = walls / ((COLS - 1) * ROWS + (ROWS - 1) * COLS);

      let s = 1 - Math.abs(density - 0.40) * 2.4;

      let open = 0;
      for (let r = 1; r < ROWS - 1; r++) {
        for (let c = 1; c < COLS - 1; c++) {
          if (this.openSides(r, c) >= 3) open++;
        }
      }
      if (open < 30) s -= (30 - open) * 0.012;
      if (s < 0) s = 0;
      return s;
    }

    buildWalls() {
      const walls = [];
      const W = this.w, H = this.h;

      /* 四周外墙 */
      walls.push({ x: 0, y: 0, w: W, h: T });
      walls.push({ x: 0, y: H - T, w: W, h: T });
      walls.push({ x: 0, y: 0, w: T, h: H });
      walls.push({ x: W - T, y: 0, w: T, h: H });

      /* 竖墙：同一列上连续的段合并 */
      for (let c = 1; c < COLS; c++) {
        let r = 0;
        while (r < ROWS) {
          if (!this.vAt(r, c)) { r++; continue; }
          let r2 = r;
          while (r2 + 1 < ROWS && this.vAt(r2 + 1, c)) r2++;
          const y0 = r * CELL - T / 2;
          const y1 = (r2 + 1) * CELL + T / 2;
          walls.push({ x: c * CELL - T / 2, y: y0, w: T, h: y1 - y0 });
          r = r2 + 1;
        }
      }

      /* 横墙：同一行上连续的段合并 */
      for (let r = 1; r < ROWS; r++) {
        let c = 0;
        while (c < COLS) {
          if (!this.hAt(r, c)) { c++; continue; }
          let c2 = c;
          while (c2 + 1 < COLS && this.hAt(r, c2 + 1)) c2++;
          const x0 = c * CELL - T / 2;
          const x1 = (c2 + 1) * CELL + T / 2;
          walls.push({ x: x0, y: r * CELL - T / 2, w: x1 - x0, h: T });
          c = c2 + 1;
        }
      }

      this.walls = walls;
    }

    /* ---------------- 碰撞 ---------------- */

    solidAt(x, y) {
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        if (x >= w.x && x <= w.x + w.w && y >= w.y && y <= w.y + w.h) return true;
      }
      return false;
    }

    resolveCircle(e, r) {
      let touched = false;

      for (let pass = 0; pass < 2; pass++) {
        let hit = false;
        for (let i = 0; i < this.walls.length; i++) {
          const w = this.walls[i];
          if (e.x + r < w.x || e.x - r > w.x + w.w ||
              e.y + r < w.y || e.y - r > w.y + w.h) continue;

          const nx = U.clamp(e.x, w.x, w.x + w.w);
          const ny = U.clamp(e.y, w.y, w.y + w.h);
          let dx = e.x - nx, dy = e.y - ny;
          const d2 = dx * dx + dy * dy;
          if (d2 >= r * r) continue;

          if (d2 > 1e-8) {
            const d = Math.sqrt(d2);
            const push = r - d;
            e.x += (dx / d) * push;
            e.y += (dy / d) * push;
          } else {
            const left = e.x - w.x, right = w.x + w.w - e.x;
            const top = e.y - w.y, bottom = w.y + w.h - e.y;
            const m = Math.min(left, right, top, bottom);
            if (m === left) e.x = w.x - r;
            else if (m === right) e.x = w.x + w.w + r;
            else if (m === top) e.y = w.y - r;
            else e.y = w.y + w.h + r;
          }
          hit = true;
          touched = true;
        }
        if (!hit) break;
      }

      return touched;
    }

    raycast(x0, y0, x1, y1, radius = 0) {
      const dx = x1 - x0, dy = y1 - y0;
      if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null;

      let best = null;
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        const hit = U.rayBox(x0, y0, dx, dy,
          w.x - radius, w.y - radius, w.x + w.w + radius, w.y + w.h + radius);
        if (!hit) continue;
        if (!best || hit.t < best.t) best = hit;
      }
      return best;
    }

    hasLOS(x0, y0, x1, y1) {
      return !this.raycast(x0, y0, x1, y1);
    }

    /* BFS（以单元格为节点，墙在边界上） */
    findPath(sx, sy, tx, ty) {
      const sc = U.clamp(Math.floor(sx / CELL), 0, COLS - 1);
      const sr = U.clamp(Math.floor(sy / CELL), 0, ROWS - 1);
      const tc = U.clamp(Math.floor(tx / CELL), 0, COLS - 1);
      const tr = U.clamp(Math.floor(ty / CELL), 0, ROWS - 1);
      if (sc === tc && sr === tr) return [{ x: tx, y: ty }];

      const total = ROWS * COLS;
      const prev = new Int32Array(total).fill(-2);
      const queue = new Int32Array(total);
      const start = sr * COLS + sc;
      const goal = tr * COLS + tc;
      let head = 0, tail = 0;
      prev[start] = -1;
      queue[tail++] = start;
      let found = false;

      while (head < tail) {
        const cur = queue[head++];
        if (cur === goal) { found = true; break; }
        const r = (cur / COLS) | 0;
        const c = cur % COLS;

        if (c < COLS - 1 && prev[cur + 1] === -2 && !this.vAt(r, c + 1)) {
          prev[cur + 1] = cur; queue[tail++] = cur + 1;
        }
        if (c > 0 && prev[cur - 1] === -2 && !this.vAt(r, c)) {
          prev[cur - 1] = cur; queue[tail++] = cur - 1;
        }
        if (r < ROWS - 1 && prev[cur + COLS] === -2 && !this.hAt(r + 1, c)) {
          prev[cur + COLS] = cur; queue[tail++] = cur + COLS;
        }
        if (r > 0 && prev[cur - COLS] === -2 && !this.hAt(r, c)) {
          prev[cur - COLS] = cur; queue[tail++] = cur - COLS;
        }
      }
      if (!found) return null;

      const path = [];
      let cur = goal;
      let guard = 0;
      while (cur !== start && guard++ < total) {
        const r = (cur / COLS) | 0;
        const c = cur % COLS;
        path.push({ x: c * CELL + CELL / 2, y: r * CELL + CELL / 2 });
        cur = prev[cur];
        if (cur < 0) break;
      }
      path.reverse();
      if (path.length) path[path.length - 1] = { x: tx, y: ty };
      return path;
    }

    /* ---------------- 静态地表 ---------------- */

    renderStatic(dpr) {
      const W = this.w, H = this.h;
      const cv = document.createElement('canvas');
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
      const ctx = cv.getContext('2d');
      ctx.scale(dpr, dpr);

      /* 地面：原版就是一块干净的浅灰 */
      ctx.fillStyle = P.floor;
      ctx.fillRect(0, 0, W, H);

      /* 极淡的网格线，帮助判断距离与反弹角度 */
      ctx.strokeStyle = 'rgba(20,20,20,.040)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = 1; c < COLS; c++) {
        const x = Math.round(c * CELL) + 0.5;
        ctx.moveTo(x, 0); ctx.lineTo(x, H);
      }
      for (let r = 1; r < ROWS; r++) {
        const y = Math.round(r * CELL) + 0.5;
        ctx.moveTo(0, y); ctx.lineTo(W, y);
      }
      ctx.stroke();

      /* 墙体底板（黑色描边）：先整体铺黑，再压上灰色填充 */
      const OUT = 2.5;
      ctx.fillStyle = P.wallEdge;
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        ctx.fillRect(w.x - OUT, w.y - OUT, w.w + OUT * 2, w.h + OUT * 2);
      }

      ctx.fillStyle = P.wall;
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        ctx.fillRect(w.x, w.y, w.w, w.h);
      }

      /* 顶部一道极淡的受光，让墙不至于死板 */
      ctx.fillStyle = 'rgba(255,255,255,.10)';
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        ctx.fillRect(w.x, w.y, w.w, 2);
      }

      ctx.fillStyle = 'rgba(0,0,0,.10)';
      for (let i = 0; i < this.walls.length; i++) {
        const w = this.walls[i];
        ctx.fillRect(w.x, w.y + w.h - 2, w.w, 2);
      }

      return cv;
    }
  };
})();
