# DeepSeek Arena Policy · Tank Trouble

一个**自包含、无外部依赖**的 Tank Trouble 策略，目标是在 8 车混战 / 1v1 中尽可能拿到**回合胜场**（最后存活者获胜），而不是单纯刷击杀。

> 本目录之外的游戏代码只读不改；按要求没有读取 `policy/` 下其他 AI 的策略文件。

---

## 文件

| 文件 | 说明 |
|---|---|
| `policy.js` | 主策略。加载后挂到 `TT.deepseekPolicy`，同时兼容多种 Arena 调用方式。 |
| `test/sim.mjs` | Node 无头对局模拟器：加载游戏原始 `util/map/entities/ai` 与本策略，跑完整物理回合。 |
| `test/variants.json` | 测试台使用的参数扫描列表（可自行增删）。 |

运行测试：

```bash
node policy/deepseek/test/sim.mjs --rounds 200 --mode ours --seat 0 --maxTime 60
node policy/deepseek/test/sim.mjs --rounds 100 --mode seats
node policy/deepseek/test/sim.mjs --rounds 200 --mode duel --seat 0
node policy/deepseek/test/sim.mjs --rounds 0 --mode match --matches 40 --roundsToWin 5
node policy/deepseek/test/sim.mjs --mode sniper --seat 0        # 0 抖动的完美瞄准 bot 压力测试
```

---

## 接入 Arena

在游戏脚本（`util.js / map.js / entities.js / ai.js`）之后加载本文件：

```html
<script src="policy/deepseek/policy.js"></script>
```

推荐为每辆坦克创建一个 brain，这样多辆坦克共用同一份策略也互不串状态：

```js
// 开场：给坦克挂 brain
const brain = TT.deepseekPolicy.create(tank, game);
// 每帧：返回 { drive, steer, shoot }
const input = brain(game, dt);
// 也接受：brain.update(tank, game, dt) / brain.think(tank, game, dt)
tank.update(dt, input, game);
```

也可以直接在每帧调用全局入口：

```js
const input = TT.deepseekPolicy(tank, game, dt);
// 或 TT.deepseekPolicy.update(tank, game, dt)
// 或 TT.deepseekPolicy.think / getInput / getAction / decide / step
```

兼容的调用形态：

- `(tank, game, dt)`、`(game, tank, dt)`、`({ tank, game, dt })`
- 每个 brain 可当函数调用，也可用 `.update / .think / .getInput / ...`
- 没传 `dt` 时，会从 `game.time` 的差分自动推算；否则回落到 `1/60`
- `TT.deepseekPolicy.attach(tank, game)` 会把 brain 同时挂到 `tank.policy` 与 `tank.brain`
- `TT.deepseekPolicy.install()` 可把 `TT.updateBot` 直接替换成本策略（用于单策略整局测试）

每辆坦克的状态用 `WeakMap` 按坦克对象隔离，并在以下情况自动重置：

- 地图对象变化 / `game.roundSerial`（或 `game.round`）变化
- 坦克位置瞬移超过 300px（部分 Arena 会复用坦克对象做重生）

函数内部全程 `try/catch`，异常时返回 `{drive:0, steer:0, shoot:false}`，不会打断游戏循环。

---

## 策略设计

### 1. 首要目标：活着

游戏规则是「最后存活者赢下回合」，击杀只进击杀榜。因此策略把生存放在第一位：

- 开火前完整模拟子弹弹道，确认它不会反弹回到自己身上；
- 任何会回到自己 30px 内的射击一律放弃；
- 同时在场自弹默认最多 2 发，避免弹幕反噬；
- 所有在飞子弹（包括自己的）都进入威胁列表，用短程 MPC 避弹。

### 2. 精确落角射击

坦克转向是 `3.3 rad/s`，按帧离散。策略用**分数转向**在最后一步精确落到目标角：

```js
steer = angleDiff / (turnRate * dt)
```

只要目标角在一帧步长内，就能一帧转到位并开火，不会出现过冲/抖动。射击解算包含：

- 目标速度提前量，迭代求命中时间（最多 3 次）
- 检查炮口到预判点的整段射线
- 炮口是否卡墙、是否会打中墙
- 目标预判点落墙时回退到当前位置

### 3. 避弹 MPC

每帧对附近每颗子弹用与游戏一致的反射逻辑模拟最多约 1.15 秒，计算它和自身的最近接近时间/距离。若有子弹会在约 1 秒内命中：

- 枚举「转向 × 油门」候选动作（前进/倒车/停车 × 5 档转向）
- 用与 `Tank.update` 一致的加速度、倒车、刹车、墙体碰撞模型推演 0.3~0.9 秒
- 选「与所有威胁子弹的最小距离最大」的动作
- 带碰撞惩罚和动作惯性奖励，避免左右抖动

### 4. 台球反弹射击（贡献最大的机制之一）

没有直射视线时，枚举每面墙的 4 个表面，把目标点镜像到墙后，求精确反弹点：

- 检查反弹点确实落在该墙面内
- 检查「炮口→反弹点」与「反弹点→目标」两段都无遮挡
- 目标会动，用不动点迭代求命中时刻
- 严格自伤检查后才扣扳机

消融测试（禁用反弹射击）胜率从 **41.5% 掉到 25.0%**，因此默认开启。

### 5. 单入口伏击营地

开局在世界里选一个「开口少、离敌人约 650px」的格子作为营地：

- 优先 1~2 个开口的死角/走廊，降低被侧翼包抄的概率
- 到达后原地守 5.5 秒，炮口朝向最可能来人的入口
- 敌人接近到 300px 以内、或场上只剩 3 人时，转为主动猎杀

### 6. 目标选择：优先打「正在和别人交火」的敌人

- 对正在瞄准我的敌人：**远距离扣分，300px 内才优先处理**
- 对正在和第三个坦克交火的敌人：加分（趁乱收人头）
- 保留目标粘性，避免准星来回切换

### 7. 残局

- 只剩 3 人及以下时主动清场
- 追击只在 26px 内停车，只要出现视线立即交火
- 残局中不主动拒绝开火，但对换命极其谨慎（自伤检查仍全程生效）

### 8. 其他细节

- 被多个敌人同时看见时，会尝试移动到能断开所有相关视线的近点（掩体）
- 装填期间默认不做大幅预扭（数据上略微降低胜率，已关闭），但保留「被瞄准且无法还手时」的掩体/扭动能力
- 卡住 0.6 秒后自动倒车脱困并清空路径
- 路径跟随带目标变化检测（目标移动 >60px 立即重算），避免追着旧路径跑

---

## 可调参数

所有参数都有默认值，可在加载后覆盖：

```js
TT.deepseekPolicy.setParams({
  fightClose: 300,
  fightFar: 650,
  campOpenBias: 6,
  maxLive: 2
});
```

主要参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `campDist` | 650 | 营地与最近敌人的理想距离 |
| `campMinOpens` | 1 | 允许营地的最少开口数 |
| `campOpenBias` | 6 | 开口越少加分越多 |
| `guardHold` | 5.5 | 营地驻守秒数 |
| `guardExitDist` | 300 | 敌人近于此距离转猎杀 |
| `guardExitAlive` | 3 | 存活数 ≤ 此值转猎杀 |
| `fightClose` | 300 | 近于此距离倒车拉开 |
| `fightFar` | 650 | 远于此距离推进 |
| `maxLive` | 2 | 同时在场的自弹上限 |
| `aimWeight` | -3 | 目标正瞄准我时的权重（负值=回避） |
| `engagedBonus` | 2 | 目标正与别人交火时的权重 |
| `threatDist` | 300 | 此距离内仍优先处理瞄准我的敌人 |
| `bankShots` | 1 | 是否使用反弹射击 |
| `coverUnderFire` | 1 | 被瞄准时是否找掩体 |
| `preJink` | 0 | 装填期预扭（默认关） |

`TT.deepseekPolicy.getParams()` 可查看当前值。

---

## 实测数据

测试条件：游戏原始物理、60Hz、两张固定地图、每回合随机出生点；对手为游戏内置 bot（`js/ai.js`）。

### 8 人混战（每回合最后存活者胜）

| 指标 | 结果 |
|---|---|
| 单回合胜率（8 个座位各 80 回合，共 640 回合） | **平均 44.7%**（随机期望 12.5%；座位区间 37.5%~55%） |
| 进入决赛 2 人 | 59.1% 的回合 |
| 决赛 2 人胜率 | 75.5% |
| 自杀（640 回合） | **1 次** |
| 60 秒人工截断（640 回合） | 10 次（1.6%；真实游戏无回合时限，会继续打到胜者产生） |

座位 0 的 80 回合样本：**48.8%** 回合胜率、决赛胜率 86.7%、0 自杀。

### 赛制（先到 5 胜）

| 场景 | 夺冠率 |
|---|---|
| 8 人混战，60 场（40 + 20 两批） | **90.0%**（54/60） |
| 8 人混战 vs 7 个 0 抖动完美瞄准 bot，20 场 | **90.0%**（18/20） |
| 1v1 vs 内置 bot，60 场（座位 0 / 1） | **96.7% / 98.3%** |

### 真实浏览器冒烟测试

在 `index.html` 中动态加载本策略，让 0 号坦克使用它、其余为内置 bot：

- 连续 3 场「先到 5 胜」全部由本策略夺冠（对手最高分均为 3）
- 控制台 0 报错
- 4993 次 `game.update(1/60)` 仅消耗约 1.1 秒 CPU（8 车，含 7 个内置 bot）

### 压力测试

- 1v1 vs 完美瞄准 bot：单回合 **81.3%** 胜率（150 回合，0 自杀）
- 8 人 vs 7 个完美瞄准 bot：单回合 36.7%，先到 5 胜赛制 90%
- 1v1 vs 内置 bot：单回合 83.3%（座位 0）/ 76.7%（座位 1），先到 5 胜 96.7% / 98.3%

---

## 注意事项

- 本策略**不硬编码地图**，全部通过 `game.map` 的运行时接口（`raycast / solidAt / hasLOS / findPath / openSides`）工作，两张固定地图和未来地图都能用。
- 不修改任何游戏对象，只返回输入；`WeakMap` 状态按坦克隔离。
- 没有读取、依赖或针对其他 AI 的策略文件；以上对手均为内置 bot 或内置 bot 的完美瞄准变体。
- 极端情况下（双方都极度谨慎）1v1 可能拖很久；策略在残局会主动追击，但若 Arena 设置较短的单回合时限，可能产生无胜者回合。
- 若 Arena 的调用约定与上述不同，只需在 Arena 适配层把输入输出映射成 `{drive, steer, shoot}`；策略本体不需要改。
