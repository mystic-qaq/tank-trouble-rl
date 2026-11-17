# GLM Tank Trouble 竞技策略

`glm_policy.js` 是一个自包含的 Tank Trouble AI 策略，接口与游戏内置的
`TT.updateBot(tank, game, dt)` 完全同签名，返回 `{ drive, steer, shoot }`。

## 策略概要

核心思路：**击杀 +1、自杀 −1、活到最后 = 胜场 → 生存第一，出手必中**。

四大能力（相对内置 bot）：

| 能力 | 说明 |
| --- | --- |
| 弹道预知 | 子弹物理完全确定，首见即模拟整条反弹轨迹并缓存（WeakMap），预知每颗子弹未来约 1.3s 的路径，自己的流弹同样纳入威胁 |
| 主动规避 | 被瞄准时对 9 组机动（转向 × 油门）做 0.6s 滚动模拟（含碰撞推挤），选存活期望最高的机动 |
| 反弹猎杀 | 72 向快速弹道扫描（DDA 网格步进 + 敌人位移外推），无直瞄也能绕墙击杀；先弹回自己的弹道自动剔除；僵局时朝「掠过敌人身边」的角度打骚扰流弹 |
| 生存走位 | BFS 距离场 + 格子评分（隐蔽性 / 与敌理想距离 / 弹道密度 / 出口数），开火后立刻侧向机动规避还击 |

另有：直瞄预判瞄准（两次迭代 + 比例转向防振荡）、压制射击、卡住自救
（位置位移检测，可捕获「顶墙空转」）、僵局打破（回合 35s 后激进度线性上升：
更贴身 / 更频开火 / 接受更长反弹弹道）。

## 接入方式（多形态兼容）

策略加载后会自动注册到以下位置（任选其一）：

```js
TT.Policies.glm       // { name:'glm', update(tank, game, dt), createController(tank) }
TT.POLICIES.glm
TT.Policy.glm
TT.GLM_POLICY
window.GLM_POLICY
window.policy
TT.registerPolicy(api)   // 若装载器提供了该函数则自动调用
```

**按坦克分配**（推荐）：装载器为每辆坦克选择策略时调用 `update(tank, game, dt)`；
或用工厂形态 `const ctrl = api.createController(tank)`，之后每帧
`ctrl(game, dt)` 返回输入。

**单策略兜底**：若 600ms 内没有任何装载器接管 `TT.updateBot`（仍是原生函数），
策略会自动接管它 —— 浏览器直接打开 `index.html` 观战即可看到效果；
装载器若自己管理 `TT.updateBot` 则不受影响。

### 引擎输入约定

- `drive`: `1` 前进（178px/s）、`-1` 倒车（117px/s）、`0` 滑行
- `steer`: `±1` 满速转向（3.3 rad/s），可为小数做比例控制
- `shoot`: `true` 时若冷却结束即开火（冷却 0.42s，场上每车最多 5 发）

策略只读公开 API（`game.map.solidAt/hasLOS/hasClearShot/raycast/findPath/
resolveCircle/vAt/hAt/openSides`、`game.tanks/bullets/kills/scores`），不修改任何游戏状态。

## 本地测试

```bash
# 无头批量对战（场景: 1v1 / duel / team4 / ffa8 / base8 / all）
cd test && node sim_test.js all 20

# 浏览器可视化验证（GLM 1/4/8 车切换 + 内置 bot 对战）
# 直接用浏览器打开 test/harness.html
```

最新成绩（先到 3 胜，各 50 场，对手为游戏内置 AI）：

| 场景 | 冠军率 | 回合胜率 | 场均击杀 | 场均自杀 |
| --- | --- | --- | --- | --- |
| 纯 1v1 | **100%** | 87.2% | 1.54 | 0 |
| 1 vs 7 混战 | **90~92%**（基线 12.5%） | ~52%（基线 12.5%） | 5.5~6.1 | 0 |
| 4 vs 4 | **100%** | 96.8% | 20.0 | ~0 |

超时死锁率与内置 bot 基线持平（回合无时限，内置 bot 也会互卡）。

## 文件

- `glm_policy.js` — 策略本体（自包含，无依赖顺序要求）
- `test/sim_test.js` — Node 无头对战统计
- `test/harness.html` — 浏览器可视化验证
