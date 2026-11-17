# Tank Trouble · 统一竞技场

从 `policy/` 下的 7 份 AI 策略与游戏内置 Bot 中任选 **2–8 个**同场对战，
使用真实引擎物理，统计各自的回合胜场、击杀与自毁。

当前版本修复了炮口越墙生成子弹、反弹折线命中判定，以及观战缩放/计时/策略分发问题。
仓库原有 `report*.json` / `report.html` 和下文历史实验使用旧引擎，不代表修复后的排名；请重新运行统计。

> 设计红线：**不改任何一份策略源码**。所有适配都发生在竞技场侧。

---

## 一、文件

| 文件 | 作用 |
| --- | --- |
| `arena.mjs` | 核心：沙箱装载、可选参赛名单、出生布局配平、单局对局 |
| `worker.mjs` | 并行工作线程（每线程跑一批局） |
| `run.mjs` | 主入口：跑 N 局 → 统计 → 打印表格 → 写 `report.json` |
| `validate.mjs` | 集成校验：每个策略单独 vs 7 内置 Bot，跟各家 README 自测成绩对账 |
| `validate-worker.mjs` | 校验用的工作线程 |
| `metric-check.mjs` | 用 minimax 自带测试台算出「真实每回合胜率」，解释口径差异 |
| `timeout-probe.mjs` | 把超时局用更长时间重跑，判断是真僵死还是分得慢 |
| `audit.mjs` | **作弊审计**：Proxy 拦截策略对游戏状态的读写 |
| `steer-audit.mjs` | 审计各策略的 steer 有没有超出物理转向上限 |
| `head2head.mjs` | 逐对单挑（28 对 × 互换座位），交叉验证排名 |
| `seat-check.mjs` | 座位中性检验：每个策略吃到多少座位红利 |
| `build-vendor.mjs` | 给浏览器页生成 ESM 策略的 classic script 版（写到 `vendor/`） |
| `index.html` / `spectator.js` / `spectator.css` | 浏览器观战、名单选择及累计排行榜 |
| `schedule.js` | 浏览器和无头模式共用的轮转与出生种子计划 |
| `regression.mjs` | 碰撞和 2–8 人出生布局配平回归检查 |
| `report.html` | 战报页（由 `report.json` 生成） |
| `bench.mjs` | 单核吞吐标定（先跑它再定总预算） |
| `sim/` | **大规模模拟协议**：样本量计算、三层实验矩阵、断点续跑、聚类 bootstrap + PL 分析，见 `sim/README.md` |

---

## 二、怎么跑

```bash
# 208 局（默认），局数为 16 的倍数，两张图及 8 个席位完整配平
node --experimental-vm-modules arena/run.mjs

# 自定义
node --experimental-vm-modules arena/run.mjs --rounds 208 --seed 20260101 --jobs 8

# 任选几个 AI；三人对局建议局数取 6 的倍数
node --experimental-vm-modules arena/run.mjs --players gpt,hy,minimax --rounds 60 --out arena/report-selected.json

# 物理与排程回归
node arena/regression.mjs

# 集成校验：每个策略 vs 7 内置 Bot，跟 README 自测成绩对账
node --experimental-vm-modules arena/validate.mjs --rounds 40

# 浏览器观战：直接双击 arena/index.html
```

> `--experimental-vm-modules` 是必需的：`kimi/policy.mjs` 和 `minimax/policy.js`
> 是 ES Module，竞技场用 `vm.SourceTextModule` 把它们载进同一个沙箱上下文。
> 用这个 flag 而不是改写成 classic script，是为了**保证策略源码一字不动**。

---

## 三、装载方式（关键适配点）

沙箱里先载入游戏本体 `js/{util,map,entities,ai,game}.js`，再逐个载入策略：

| 策略 | 形式 | 装载方式 | 取出入口 |
| --- | --- | --- | --- |
| deepseek | classic | `vm.runInContext` | `TT.deepseekPolicy` |
| glm | classic | `vm.runInContext` | `TT.Policies.glm.update` |
| gpt | classic | `vm.runInContext` | `TT.policies.gpt` |
| hy | classic | `vm.runInContext` | `TT.Policies.hy` |
| qwen | classic | `vm.runInContext` | `TT.policies.qwen.update` |
| minimax | **ESM** | `vm.SourceTextModule` | `default` 导出 |
| kimi | **ESM** | `vm.SourceTextModule` | `default` 导出 |
| 内置 Bot | — | 引擎原生 `TT.updateBot` | — |

三处必须注意的细节：

1. **内置 Bot 原件要在载入任何策略之前抓住。**
   `policy/gpt/ai.js` 在加载时会把 `TT.updateBot` 覆盖成它自己。
   竞技场在加载策略**之前**先存下引擎原版 `TT.updateBot`，作为内置 Bot 席位。

2. **`setTimeout` 被桩成空函数。**
   `policy/glm/glm_policy.js` 有一段「600ms 后若无人接管 `TT.updateBot` 就自己接管」的
   兜底逻辑。竞技场自己接管 `TT.updateBot`，所以这段兜底必须屏蔽，否则 GLM 会抢走整张竞技场。

3. **ESM 策略看到的 `globalThis` 就是沙箱全局。**
   `vm.SourceTextModule` 传 `context: ctx`，所以 `minimax` 里的 `G.TT.CFG` 能读到沙箱的
   `TT.CFG`（已实测确认）。

### 输入归一化

竞技场接管 `TT.updateBot`，按座位分发到对应策略，并把返回值统一成 `{drive, steer, shoot}`：

- `drive` 夹到 `[-1, 1]`（引擎只取符号，无副作用）；
- `steer` **原样透传、不截断** —— deepseek / hy / minimax 用分数转向做「一帧精确落角」，
  截断会直接破坏它们的瞄准；
- `shoot` 转布尔；
- 任何异常都吞掉并计一次错误，返回 `{drive:0, steer:0, shoot:false}`。

---

## 四、公平性设计

### 出生点与席位配平

原 `pickSpawns()` 把贪心选点的先后顺序直接当作玩家编号，会让先选的点和后选的点具有不同的统计分布。
现在选完整组后再用传入随机源做 Fisher–Yates 洗牌，避免把选点顺序优势固定给 P1。

仅轮转策略编号仍不够：每局都换出生点，有限样本中无法保证吃到相同布局。
`arena/schedule.js` 因此按参赛人数 N 分组，每组 N 局：

- 同组使用同一地图和独立的出生随机种子，出生坐标与朝向完全相同；
- 策略逐局轮换，每个策略恰好使用每一个出生席位一次；
- 下一组换一套出生布局，地图 A/B 按组交替；
- 出生随机源独立于策略和特效随机数，不会被策略调用次数影响。

局数取 **2 × N 的倍数**时，两张地图、每张地图上的每个出生布局都能完整轮换。
默认 208 局适合 8 人；三人可用 60 局，两人可用 40 局。未完成整组会在命令行提示。

不对称地图中的位置本身仍可能有不同的生存价值；这里保证各 AI 获得相同的出生机会，
不保证每个地理位置或不同实力策略的胜率相同。策略随机决策也仍会造成结果波动。

### 局面隔离

**每一局都新建一个完整沙箱**（重新载入引擎 + 全部 7 份策略），并给该局一个独立随机种子
`roundSeed(baseSeed, roundIndex)`。因此：

- 策略的内部状态（各家的 WeakMap / Map 缓存）不会跨局串味；
- 同一个 `(baseSeed, roundIndex, players)` 永远得到同一局，可复现；
- 并行跑和串行跑结果完全一致。

---

## 五、浏览器观战页

直接双击 `arena/index.html` 即可（无需构建、无需联网、无需起服务器）。

- 右侧勾选 2–8 个 AI，点击「开始新对战」应用名单并清零统计；
- 默认 1×，使用真实帧间隔推进固定 1/60 秒物理步，支持 0.25×–16×；
- 暂停后不会跳到下一局，单步按钮推进一帧；已结束时单步进入下一局；
- 排行榜按策略身份累计胜场/击杀/自毁，轮换席位不改变归属，结果自动展示 2.25 秒；
- 名称和颜色跟随策略，边缘名称自动留在地图内；小屏时排行榜移到画面下方；
- 回合计时每局清零，60 秒超时记平局；重开本局保留已有累计统计。

浏览器在载入 GPT 策略前保存引擎原始 Bot，再按实际轮转表分发策略。

浏览器页有个额外问题要处理：`kimi` 和 `minimax` 是 ES Module，
而 **`file://` 协议下 `import()` 动态加载 ES module 会被 Chrome 直接拒绝**
（`Failed to fetch dynamically imported module`），双击打开就载入不了这两个策略。

解法是 `arena/build-vendor.mjs`：把 ESM 源码做**机械转换**（去掉 `export` 关键字、
整体包一层 IIFE、尾部挂到 `window`），输出到 `arena/vendor/`：

```
policy/kimi/policy.mjs    → arena/vendor/kimi.classic.js
policy/minimax/policy.js  → arena/vendor/minimax.classic.js
```

两点保证：

- **`policy/` 一个字节都不动**，vendor 版只是浏览器用的副本；
- minimax 的 vendor 版会跟官方 `policy/minimax/policy.global.js` 做**代码本体交叉校验**
  （去注释后逐字节比对），确保转换逻辑没走样。

> 包 IIFE 是必须的：classic script 的顶层 `const`/`function` 会进全局作用域，
> 而 minimax 和 kimi **都声明了 `const TAU`**，同时加载会直接抛
> `Identifier 'TAU' has already been declared`。

改完策略后重新生成 vendor 版：

```bash
node arena/build-vendor.mjs
```

浏览器页通过 Playwright 检查：60/120/144 Hz 的 1× 计时、慢放、暂停、名单校验、自动续局、统计清零及桌面/小屏布局。
无头模式另实测两人 4 局、三人 6 局、八人 16 局，策略异常与异常局均为 0。

---

## 六、口径说明（重要）

竞技场统计的是**真实每回合胜率**：赢下一回合 ÷ 实际打出的回合数。

这个口径和 minimax README 里那个「95.0%」**不是一回事**。用 minimax 自带的测试台
（`arena/metric-check.mjs`）可以算出：

```
实际打出的回合总数   : 566
minimax 赢下的回合数 : 150
README 口径（分母 = 场数×5×种子数 = 180）: 83.3%
真实每回合胜率（分母 = 实际回合数 566）   : 26.5%
```

minimax 的 `test/sweep8.mjs` 里 `rate = win / (matches * 5 * SEEDS.length)`，
分母假设**每场正好 5 回合**（`roundsToWin = 5`），但「先到 5 胜」的一场实际会打十几二十回合，
所以分母被严重低估、胜率被高估。

各家 README 的口径并不统一，横向对比前请注意：

| 策略 | README 里的 1v7 数字 | 分母 | 是否等同真实每回合胜率 |
| --- | --- | --- | --- |
| minimax | 95.0% | 场数×5×种子数（假设每场 5 回合） | ✗ 偏高 |
| glm | ~52% | 实际回合总数 | ✓ |
| qwen | 40.0% | 实际回合数 | ✓ |
| deepseek | 44.7% | 实际回合数（8 座各 80 回合共 640） | ✓ |
| hy | 54.0% | 实际回合数 | ✓ |

`arena/validate.mjs` 的作用就是**用统一口径重测**，看各家在自己的老对手面前还剩多少实力。

---

## 七、超时处理

正式游戏、浏览器 Arena、无头 Arena 和 RL 训练统一使用 **60 秒交战时间**上限，
由 `TT.CFG.ROUND_TIME_LIMIT` 和共用引擎判定。超时记为**平局**，无人获得胜场，已经发生的击杀保留。
下文 90/300 秒实验为旧版本历史结果。

实测 200 局里有 9 局超时。用 `arena/timeout-probe.mjs` 把这 9 局用 300 秒重跑，
结果 **8 局到 300 秒仍然僵死**（只有 1 局在 115 秒分出胜负）——
说明这些是真的「谁都不肯先动手」僵局，不是跑得慢。

僵局的特征很明显：最后活着的都是**偏防守**的策略（HY / Kimi / DeepSeek），
说明「生存优先」的打法在混战里容易互相顶牛。这一点在解读胜率时要留意。

---

## 八、「HY 胜率最高是不是作弊」

结论：**没有作弊**。做了 4 项独立、可证伪的检验。

### 1. 运行时写入审计（`arena/audit.mjs`）

用 Proxy 包住 `game` / `tank` / `bullet`，只在**策略被调用的那一瞬间**记录所有写操作：

| 策略 | 写操作 | 非公开读取 |
| --- | --- | --- |
| deepseek / glm / gpt / hy / kimi / minimax / qwen | **0 次** | 无（deepseek 读 `tank.tank/tanks`、hy 读 `tank.heading/dir/...`、`game.walls/world`，都是它为跨引擎兼容写的多字段兜底，引擎里不存在 → 读到 undefined） |
| 内置 Bot | 4 次（`tank.ai`） | `tank.barrelTip` |

内置 Bot 写的 `tank.ai` 是引擎原生的记忆槽（`js/ai.js` 自己就这么干），不是作弊。
**7 份 AI 策略全程零写入。**

### 2. steer 超物理审计（`arena/steer-audit.mjs`）

竞技场刻意不截断 `steer`（deepseek/hy/minimax 用分数转向做一帧精确落角，截断会毁掉瞄准）。
但引擎是 `angle += steer * T.turn * dt`，所以 `|steer| > 1` 意味着**转向超过物理上限 3.3 rad/s**。

实测：**8 个策略的 steer 最大值都恰好是 1.00，`|steer|>1` 占比全为 0.00%**。
不截断没有给任何一方好处。

### 3. 座位中性检验（`arena/seat-check.mjs`，3 种子 600 局）

- 座位边际胜率 8.0%~14.7%（公平线 11.8%，最大偏离 3.8 个点）
- **每个策略的座位红利都恰好 ±0.0 个点** —— 轮转完美配平
- **HY 在 8/8 个座位上都是第 1 名**（25.3%~45.3%）

如果优势来自「幸运座位」，它在最差座位上就该掉队，实际没有。

### 4. 决定性对照：逐对单挑（`arena/head2head.mjs`，28 对 × 30 局 × 互换座位 = 1680 场）

```
GPT Sentinel 72.1% > DeepSeek 69.2% > HY 68.1% > GLM 64.9%
> Qwen 45.3% > Kimi 36.8% > minimax 27.1% > 内置 Bot 17.5%
```

**HY 单挑只排第 3**。如果它靠作弊取胜，作弊手段在 1v1 里同样有效 —— 但它没有。
（HY 在单挑里平手 22 局最多、平均时长 15.0s 最长，也是「苟住」的表现。）

### 那 HY 混战第一的真实机制是什么？

**是打法契合赛制，不是作弊。** 游戏规则是「最后存活者赢」，所以混战里**生存率就是胜率**：

| 策略 | 胜率 | 决赛率 | 存活到末 | 击杀/局 |
| --- | --- | --- | --- | --- |
| HY | 33.2% | **51.8%** | **37.7%** | 1.17 |
| GPT Sentinel | 21.2% | 43.8% | 21.5% | **1.79** |
| GLM | 18.2% | 38.0% | 19.0% | 0.99 |
| DeepSeek | 15.2% | 27.3% | 18.3% | 0.93 |
| Qwen | 5.0% | 20.7% | 5.0% | 0.28 |

- **HY 是「防守型」**：赢得多但杀得少（1.17 击杀/局，GPT 是 1.79）
- **GPT 是「进攻型」**：杀得最多，但自己死得也多
- 33 局超时里 **HY 存活 27 次最多**（双方都太保守，谁都不肯先动手 → 真僵局）

一句话：**按「胜场」排，HY 第一；按「击杀」排，GPT 第一；按 1v1 单挑排，GPT 第一。**
换个赛制冠军就换人 —— 这本身就说明排名反映的是打法与规则的匹配度，而不是某个策略在偷看答案。

