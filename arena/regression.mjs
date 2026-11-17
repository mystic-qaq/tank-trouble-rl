/* node arena/regression.mjs — 真实引擎的物理与公平排程回归。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import './schedule.js';

const context = vm.createContext({ console, Math, window:{}, TT:{} });
context.window = context;
for (const file of ['util', 'map', 'entities', 'game']) {
  vm.runInContext(fs.readFileSync(new URL('../js/' + file + '.js', import.meta.url), 'utf8'), context);
}
const { TT } = context;
TT.Audio = { shoot(){}, bounce(){}, explode(){} };
const fx = { fizzle(){}, muzzle(){}, bounceSpark(){}, explosion(){} };
function world(walls, tanks = []) {
  const map = Object.create(TT.GameMap.prototype);
  map.walls = walls;
  return { map, tanks, bullets:[], fx, state:'playing', shake(){},
    killTank(tank) { tank.alive = false; } };
}
function tank(x, y, index = 1, radius = 16) { return { x,y,index,radius,alive:true }; }
function bullet(x, y, angle = 0) { return new TT.Bullet(x,y,angle,0,'#fff'); }

// 炮口在墙另一侧时不能直接生成一发穿墙弹。
const wall = { x:100, y:0, w:10, h:200 };
const shooter = new TT.Tank(0, { x:84, y:100, a:0 }, TT.COLORS[0], true);
const blocked = world([wall]);
shooter.fire(blocked);
assert.equal(blocked.bullets.length, 0, 'barrel crossing a wall must fizzle');
const open = world([]);
shooter.fire(open);
assert.equal(open.bullets.length, 1, 'unobstructed shots still fire');

// 整段扫掠不能越过薄墙，包含弹体半径与高速移动。
const b = bullet(60,100);
b.update(0.1, world([wall]));
assert.equal(b.bounces, 1);
assert.ok(b.x < 100 - b.radius);
const grazing = bullet(60, -2);
grazing.update(0.1, world([wall]));
assert.equal(grazing.bounces, 1, 'bullet edge must collide near wall endpoints');
const behind = tank(130,100);
bullet(60,100).update(0.1, world([wall], [behind]));
assert.equal(behind.alive, true, 'wall shields a tank');

// 一帧内先命中坦克再到墙：不能在反弹后用首尾连线漏判。
const before = tank(85,100,1,2);
bullet(60,100).update(0.1, world([wall], [before]));
assert.equal(before.alive, false, 'outbound segment hits before reflecting');
const near = tank(80,100,1,2), far = tank(110,100,2,2);
bullet(60,100).update(0.1, world([], [far,near]));
assert.equal(near.alive, false);
assert.equal(far.alive, true, 'nearest tank wins regardless of array order');

// 折线的首尾连线穿过一个目标，而真实两段弹道均绕开：不能误杀。
const offPath = tank(78,112,1,1);
bullet(60,80,Math.PI / 4).update(0.14, world([wall], [offPath]));
assert.equal(offPath.alive, true, 'reflection chord is not a real trajectory');
const returnTarget = tank(74,138,1,3);
bullet(60,80,Math.PI / 4).update(0.14, world([wall], [returnTarget]));
assert.equal(returnTarget.alive, false, 'reflected segment can hit a tank');
const postRound = tank(80,100);
const over = world([], [postRound]); over.state = 'over';
bullet(60,100).update(0.1, over);
assert.equal(postRound.alive, true, 'outro cannot add kills');
assert.equal(TT.util.segCircle(0,0,1,0,0,0,10), 0);

// 2–8 人每个完整轮换组使用完全一致的出生坐标和朝向。
const ids = ['deepseek','glm','gpt','hy','kimi','minimax','qwen','builtin'];
for (let n = 2; n <= 8; n++) {
  const selected = ids.slice(0,n);
  for (let block = 0; block < 4; block++) {
    const seen = selected.map(() => new Set());
    let layout;
    for (let r = block * n; r < (block + 1) * n; r++) {
      const plan = TTArenaSchedule.plan(r, selected, 1234);
      const map = new TT.GameMap(plan.mapSeed);
      const spawns = map.pickSpawns(n, TT.util.rng(plan.spawnSeed));
      assert.equal(spawns.length,n);
      const serialized = JSON.stringify(spawns);
      if (layout) assert.equal(serialized, layout);
      layout = serialized;
      plan.seats.forEach((id,s) => seen[selected.indexOf(id)].add(s));
      for (const p of spawns) assert.equal(map.solidAt(p.x,p.y), false);
      assert.equal(new Set(spawns.map(p => p.x + ',' + p.y)).size,n);
    }
    seen.forEach(seats => assert.equal(seats.size,n));
  }
}
assert.throws(() => TTArenaSchedule.roster(['gpt']));
assert.throws(() => TTArenaSchedule.roster(['gpt','gpt']));
assert.throws(() => TTArenaSchedule.roster(['gpt','unknown']));
console.log('✓ 炮口防穿墙、弹体扫掠、反弹分段命中、最近目标、结束后不计击杀、2–8 人同布局全席位配平');
