# V8 设计契约 — Schema V3 + 架构 + 训练方案

> **状态说明：** 这是 V8 的设计契约，不是最终实验报告。实现进度、Stage-1 出口判据与 Stage-2a 安全旁路实验见根目录 `docs/03-learning-policies.md` 与 `docs/04-architecture.md`。


状态：设计定稿，待施工。
前置证据：`v7-runs/audit/`（冻结口径复测、随机基线、消融、导出对齐、闭环对齐）。

---

## 0. 三条设计公理

**公理 1 — 求解器降级为标签预言机，退出输入路径。**
输入只含物理状态与传感器读数。`traceShot` / `intercept` / `flight` / `pathsFor` /
`closest` 全部保留在**训练期损失**里作稠密监督，推理图里一个都不存在。

**公理 2 — 动作分解，不联合。**
102 类联合分布是三个已确诊故障的共同根源：开火信用被 51 个导航类稀释（实测只有
0.55% 的时间落到被选中的机动上）；提温探索随机化导航（T 0.35→2.0 使 netKills 单调
恶化到 −0.92）；sampled 模式崩溃（自杀率 0.183→0.474，直接否掉 `min(greedy,sampled)`
验收）。分解为三个独立分布后三者同时消失。

**公理 3 — 表征必须被证明能推出下一状态。**
作为显式训练目标（前向动力学辅助头），不作为架构假设。

---

## 1. Schema V3：输入

判据：一个量可作输入，当且仅当它是**状态变量的直接读取**或 **O(1) 算术重表达**
（减法、旋转、sqrt、atan2），或**对静态地图的传感器查询**。凡需对游戏动力学做
前向仿真或搜索的量，一律只能作标签。

### 1.1 每帧块 — 608 维

| 偏移 | 块 | 维 | 字段 | 引擎读取 | 归一化 |
|---|---|---|---|---|---|
| 0 | 自机状态 | 8 | x, y, sinθ, cosθ, speed, cooldown, alive, ammo | `tank.{x,y,angle,speed,cooldown,alive}`；ammo = 5 − 该 owner 活弹数 | `/W`, `/H`, —, —, `/178`, `/0.42`, —, `/5` |
| 8 | 自机上帧动作 | 3 | drive, steer, shoot | tracker 保存 | — |
| 11 | 对手状态 ×7 | 63 | 每槽 9：x, y, sinθ, cosθ, speed, cooldown, alive, presence, ammo | 同上 | 同上 |
| 74 | **对手动作 ×7** | 21 | 每槽 3：drive, steer, shoot | 见 §1.3 差分恢复 | — |
| 95 | 子弹 ×40 | 320 | 每槽 8：x, y, sinθ, cosθ, life, bounces, owner, presence | `bullet.{x,y,angle,life,bounces,owner}` | `/W`, `/H`, —, —, `/6`, `/5`, `/7`, — |
| 415 | **自我中心 lidar ×64** | 192 | 每射线 3：命中距离、墙面法向 body-frame cos/sin | `map.raycast(me, me + dir(θ+i·2π/64)·diagonal)` 返回 `{t,nx,ny}` | `t`, —, — |
| 607 | 时钟 | 1 | 1 − roundTime/60 | `game.roundTime` | — |

### 1.2 回合级块 — 346 维（每回合编码一次并缓存）

| 块 | 维 | 引擎读取 |
|---|---|---|
| 竖墙 `vFlag[10×17]` | 170 | `map.vAt(r,c)`，`VC=COLS+1=17` |
| 横墙 `hFlag[11×16]` | 176 | `map.hAt(r,c)`，`HC=ROWS+1=11` |

回合内恒定。当前 V6/V7 每帧重算 `Conv3x3×2 + pool`，一回合 3600 帧算 3600 次相同
结果——缓存是白捡的加速。

### 1.3 对手动作通道：从相邻帧状态差精确恢复，不改引擎

`Tank.update(dt,input,game)` 收到 input 但不存储；改 `js/entities.js` 会动
`source_digest` 并作废 G0。三个量都能从相邻帧精确恢复，放在观测 tracker 里算
（现成模式：`policy/rl/v2/geometry.mjs:142` 已为自机保存 `lastAngle`）：

```
steer_k = clamp( wrap(θ_k,t − θ_k,t−1) / (T.turn·dt), −1, 1 )     T.turn=3.3, dt=1/60
drive_k = sign(Δspeed_k)  仅当 |Δspeed_k| > 0.5·accel·dt，否则沿用上一帧
                                                     accel=920, brake=1150
shoot_k = (cooldown_k,t−1 > 0) ∧ (cooldown_k,t ≈ 0.42)
```

`shoot` 用 cooldown 跳变而非"是否出现新子弹"：`entities.js:93` 在炮口被堵的哑火检查
（`:101-104`）**之前**就设了 `cooldown = fireCooldown`，所以哑火尝试也能捕获。

### 1.4 已确认的非状态字段（不写入）

| 字段 | 判定 | 依据 |
|---|---|---|
| `bullet.grace` | 精确可推导 | `grace = max(0, life − 5.93)`，两者每帧减同一 dt，初值 0.07 / 6.0 |
| `tank.lastShotAge` | 死状态 | 全库只写不读 |
| `tank.recoil` | 视觉 | 只被 `draw()` 读 |
| `bullet.radius`, `tank.radius`, `bullet.speed` | 常量 | 3.2 / 16 / 620 |
| `bullet.px, py` | 引擎不读 | **但 `policy/hy/hy.mjs:592-594`、`deepseek/test/sim.mjs:264`、`kimi/test.mjs:70` 读它**。物理转移不依赖，故不入状态；若日后建模对手感知需重新评估 |
| `game.kills[]`, `scores` | 不影响动力学 | 自由混战胜负条件是"最后存活" |
| `tank.isBot` | 回合内恒定 | `game.js:286-288` 只用于选输入源 |
| 物理常数 | 网络可学 | 当前 obs 第 22/23 维是字面 `1`、`1.2`，第 28/29 维是 `T.turn*DT`、`B.speed/T.speed`——纯浪费 |

### 1.5 唯一的判断题

lidar 的 192 维是本设计里唯一落在"感知 vs 求解"边界上的东西：它是对**静态地图**的
射线查询（激光测距仪），不是对**动力学**的前向仿真。保留它的理由是归纳偏置——从
346 个墙格里让 CNN 自己学会"沿 φ 方向投射找第一面墙及其法向"是过分的请求，而反射
定律恰好需要距离和法向这两个量。**做成配置开关，砍掉即 416 维/帧。**

---

## 2. Schema V3：标签块（只在训练期存在）

### 2.1 瞄准 — 196 维

**固定基（192 维）**：64 个方向 `φ_i = θ_me + i·2π/64`，i=0..63。
这是一个**无信息的等间隔自我中心基**，不是求解器的候选集——网络必须自己找出哪个
方向好，没有任何"答案"被喂进去。对每个 φ_i 调 `traceShot(game, me, φ_i)`：

| 字段 | 维 | 类型 |
|---|---|---|
| `hit_enemy` | 64 | binary |
| `hit_self` | 64 | binary |
| `time_to_hit / 6` | 64 | continuous |

**精确伺服目标（4 维）**：由求解器的**精确候选**（`aimCandidates()`，含 `intercept()`
提前量解与镜像反弹解）算出，只作标签，精度不受 64 基限制：

| 字段 | 来源 |
|---|---|
| `sin(err_best)`, `cos(err_best)` | 最近安全候选的角误差 |
| `has_safe` | 是否存在安全候选 |
| `cone_best / π` | 该候选的角半径 |

由此生成动作标签（V7 已验证的控制律）：
```
steer* = clamp( err_best / (3.3·dt), −1, 1 )   当 has_safe，否则 = 教师 steer
drive* = 教师 drive                            走位始终学教师
fire*  = (ready ∧ has_safe ∧ |err_best| ≤ cone_best) ∨ 教师 fire
```

### 2.2 威胁 — 120 维

对 40 个子弹槽，调 `flight()` + `closest()`：

| 字段 | 维 | 归一化 |
|---|---|---|
| `ttc` | 40 | `/1.2` |
| `miss_distance` | 40 | `/diagonal` |
| `will_hit_me` | 40 | binary（`< me.radius + bullet.radius`） |

### 2.3 对手意图 — 14 维

对 7 个对手槽，由**相邻帧状态差**定义（不是求解器输出）：
```
bearing_rate_k = wrap( bearing_k,t − bearing_k,t−1 ) / dt        归一化
tracking_k     = (|bearing_rate_k| < τ) ∧ (range_k 在减小)        binary
```

### 2.4 前向状态 — 0 额外维

`s_{t+1}` 就是下一帧的 §1.1 状态块，训练时按偏移读取，不重复存储。

### 2.5 教师动作 — 3 维

`drive, steer, shoot`。

**标签块合计 333 维。** 每帧存储 `608 + 333 = 941` float32 = 3764 B；标签可用
float16 + binary 位压缩降到约 2500 B/帧。

### 2.6 求解器的已知缺陷（标签会继承）

`traceShot`（`policy/rl/v2/geometry.mjs:77-95`）的自伤判定与它自己的枪口计算不自洽：

```js
const x = me.x + Math.cos(angle)*offset, ...      // 枪口按「候选方向 angle」放置
const tvx = Math.cos(tank.angle)*tank.speed, ...  // 自机运动按「当前朝向 me.angle」外推
```

两者相差 `err`，而 `err` 正是伺服要闭合的量。修正版在 `policy/rl/v8/geometry_v8.mjs`：
自机沿「以 3.3 rad/s 转向候选方向、对齐后直行」的真实轨迹外推（圆弧段位置积分解析可解，
见 `selfPose()`），因路径不再是直线，自机碰撞改为 1/60s 采样圆重叠检测（子弹每帧
10.3px、命中半径 19.2px，不会漏判）。其他坦克仍用匀速外推——我们不知道它们的指令，
匀速是唯一诚实的假设。

**实测量级（`test_traceshot.mjs`，hy/gpt 教师 × n=2/4/8，3321 帧 / 45445 个候选）：**

| 指标 | 值 |
|---|---|
| v2 判安全 | 1882（4.14%） |
| v8 判安全 | 1905（4.19%） |
| **v2 判安全但 v8 判自伤（危险方向）** | **44，占 v2 安全判定的 2.34%** |
| v2 判不安全但 v8 判安全 | 67 |
| `hit` 判定改变（受害者身份变了） | 111 |
| **`has_safe` 翻转的帧** | **42，占 1.26%** |
| 危险方向的瞄准误差 p10 / p50 / p90 | 42.8° / **107.9°** / 155.0° |

**这个量级远小于预期，因此必须更正一条因果判断：它不是 V7 self/shot 0.055 对教师
0.016 的原因。** 理由现在很清楚：`fire_analytic` 要求 `|err_best| ≤ cone_best ≈ 2.7°`，
即**真正开火的那一刻自机已经对齐**，此时 `me.angle ≈ angle`，v2 与 v8 给出同样的答案。
所有分歧都发生在误差 42.8°–155.0° 的**未对齐候选**上，而那些候选永远不会被开火。

所以该缺陷的真实危害是**伺服选错转向目标**：在 1.26% 的帧上，`argmin|err|` 可能选中一个
实际不安全、但误差很大的候选，让自机白白转 0.6 秒（108° / 3.15°每帧）——**损失的是节奏，
不是直接的自伤**。这与 V7 实测的"打得准但出手少（2.5–3.5 发/局 对教师 7.8）"是一致的。

**但修正对 V8 比对 V7 重要得多**，必须保留：V8 的 `aimTruthBasis` 在 **64 个等间隔方向**
上生成标签，其中绝大多数远离当前朝向——正是 v2 系统性出错的区域。不修，`L_aim` 的
`hit_self` 通道会在大部分方向上教错的东西。

V7 self/shot 偏高的真正原因待查，候选有三：fire precision 0.943 意味着约 6% 的射击
不在标注安全的时刻；开火后伺服继续跟踪**移动**目标而标签只描述开火瞬间；以及
`traceShot` 对**敌方**也用匀速外推，敌人一机动，"命中"判定失效，子弹飞过去反弹回来。
V8 从奖励侧（`death_penalty` + win 目标）而非标签侧处理这一项。

`selfHorizon = 2.0s` 是自伤判定窗口（子弹寿命 6s，但 2s 后自机已远离且匀速假设退化），
是可调项。

`aimCandidates` 只考虑**最近 12 面墙**的反弹解（`geometry.mjs:100-104`），候选集是
有损启发式。V8 的 64 方向固定基不受此限（对每个方向独立 `traceShot`），是改进。

---

## 3. 动作空间：5 维，三个独立分布

```
drive ∈ {−1,0,+1}   Categorical(3)                     ← normalizeAction 用 Math.sign()
steer ∈ [−1,1]      tanh-squashed Normal(μ(z), σ(z))   ← env.mjs:8 已 clamp，tanh 直接合法
shoot ∈ {0,1}       Bernoulli(σ(z))

log π(a|z) = log π_d + log π_s + log π_f               三项独立，可分别加温
```

部署：drive 取 argmax，steer 取 μ，fire 阈值 0.5 → 确定性。

连续 steer 让 2.7° 锥宽**精确可表达**（可直接输出 `err/3.15°`），17 档量化误差消失。
探索时可**只对 fire 加温**而保持 drive/steer 锐利——这是 V7 RL 阶段被记为"需改
rollout 协议"的死结，在分解分布下是一行代码。

---

## 4. 架构

```
【回合级，缓存】
map 346 → reshape(2,11,17) → Conv3x3(2→32) SiLU Conv3x3(32→64) SiLU
   ├ global avgpool → Linear(64→256) SiLU                  map_vec (256)
   └ 空间 flatten → Linear(64→128)                          16 地图 token

【每帧】
lidar 192 → reshape(64,3) → Linear(3→64) SiLU Linear(64→128)  64 lidar token
自机 11   → Linear(11→128) SiLU                                1 自机 token（query 锚）
子弹 ×40  → 共享 Linear(8→128) SiLU                            40 子弹 token
对手 ×7   → 记忆槽输出 m_k（§5）                                7 对手 token
+ type_embedding(5→128)；padding 掩码 = presence

【融合】
tokens = [CLS] + 16 地图 + 64 lidar + 1 自机 + 7 对手 + 40 子弹 = 129 token
3 层 Pre-LN self-attention(128, 4 head, key_padding_mask)
→ LayerNorm → CLS(128) + masked-mean(128)

【主干】
z = Linear([own_tok 128, map_vec 256, CLS 128, pooled 128, 全局记忆 g 128] = 768 → 512)
    SiLU · Residual(512) · Residual(512) · LayerNorm           z (512)

【动作头 — 推理时唯一存活的部分】
drive_head  Linear(512→3)
steer_head  Linear(512→2) → μ, log σ
fire_head   Linear(512→1)

【辅助头 — 训练后全部删除，不进推理图】
aim_head     Linear(512→64×3)   §2.1 固定基
threat_head  Linear(512→40×3)   §2.2
intent_head  Linear(512→7×2)    §2.3
fwd_head     Linear(512+5→608)  公理 3：给定 (z,a) 预测下一帧输入块
```

参数量约 **4.5M**。推理图 = 编码器 + 3 个动作头，无 atan2/argmin/gather/求解器。

推理预算：浏览器 1× 速度每帧 16.6ms，V7（2.15M）实测 1.14ms。目标 **≤ 4ms**，
使 4× 回放仍流畅。地图分支缓存省掉一整条 conv 路径。

---

## 5. 记忆：按对手分槽的共享权重 GRU

```
回合开始：m_k ← 0 (k = 1..7)，g ← 0

每帧，每个对手 k：
   u_k = [ 对手状态 9 | 对手动作 3 | 相对量 4：body-frame 方位 sin/cos、距离、视线是否被墙遮挡 ]  = 16
   m_k ← GRU_shared(u_k, m_k)          # 7 槽共享同一组权重，hidden 128

每帧，全局：
   g ← GRU_global([z 摘要 | 时钟 | 存活数 | 自机动作], g)     # hidden 128

7 个 m_k 作为 7 个 token 进入融合注意力（替代原始对手 token）
```

**为什么分槽而不是单个全局 GRU**：全局 GRU 会把 7 个对手混进同一隐向量，无法回答
"**hy 在跟踪我，gpt 没有**"这种按对手索引的问题——而这正是混战里唯一重要的问题。
分槽让记忆按对手可寻址，注意力可直接 query。

**为什么共享权重**：对手在槽位间可交换（同一策略可坐任何席位），共享把参数量从 7×
降到 1×，并天然获得席位置换等变性。

**风格如何"隐式"学出**：无任何显式风格标签。GRU 在 60–120 帧（1–2 秒）窗口上累积
每个对手的 `(状态, 动作)` 序列，而 `intent_head` 与 `aim_head` 的损失迫使主干利用它
——因为"他会不会打中我"的最优预测器**必然**要建模他的射击习惯。风格表征是被任务
压力挤出来的，不是被设计的。

**不用显式世界模型做想象 rollout（Dreamer 式）**：我们有精确模拟器，想象 rollout
严格劣于真实 rollout，唯一收益是省引擎墙钟，而引擎墙钟可以用并行度买回来。所以
`fwd_head` 只作表征学习的辅助任务，不作 rollout 替代。

跨帧携带状态：8 × 128 = 1024 float。GRU 在 ONNX 里是单个算子，导出无障碍。

---

## 6. 训练

### Stage 0 — 采集

| 项 | 值 |
|---|---|
| worlds | 6000（与旧语料同量级，seed 另取） |
| 记录席位 | **hy + gpt 各一席**（见 §9 的教师选择） |
| 对手 | 其余席位从 8 个冻结策略池随机抽 |
| 预计轨迹 | ~12000 |
| 预计帧数 | ~10.8M（旧语料 7.6M） |
| 预计体积 | ~40 GB（旧 80.36 GB，新 schema 每帧 941 对 1969 float） |
| 预计耗时 | **~65 分钟**（旧采集 7.6M 帧用 1337.6s = 5692 f/s；新每帧多 64 次 `traceShot` 与 64 条 lidar，按 2× 成本估 ~2800 f/s） |
| skills 子集 | 6 个微技能场景，reference 求解器驱动，~1200 轨迹，另计 ~5 分钟 |
| 存储 | **本地盘** `$RL_ROOT/data-v3/`，不写 NFS（11-07 06:00 nfsd 挂死导致全机 60 个 D 状态进程） |
| 并行度 | **96 node worker**（256 核，`yjh` 现占约 60 核，load 15） |
| 护栏 | load average 上限 150，超限自动降 worker；RSS 监控 |
| GPU | 采集不需要 |

采集帧必须**逐帧连续存储**（旧语料已是连续的，抽帧发生在 `load_corpus` 的
`np.linspace`——新 loader 不再抽帧，因为差分与前向目标都依赖真实相邻帧）。

产出即冻结：sha256 清单 + G0 取证。

### Stage 1 — 多任务监督预训练

```
L = 1.0·L_action + 0.3·L_aim + 0.2·L_threat + 0.2·L_intent + 0.1·L_forward

L_action  = CE(drive, drive*) + SmoothL1(steer, steer*) + BCE(fire, fire*, pos_weight=8)
L_aim     = 64 方向 × [BCE(hit_enemy) + BCE(hit_self) + SmoothL1(ttc)]
L_threat  = 40 子弹 × [SmoothL1(ttc) + SmoothL1(miss_dist) + BCE(will_hit_me)]
L_intent  = 7 对手 × [BCE(tracking) + SmoothL1(bearing_rate)]
L_forward = SmoothL1(fwd_head(z,a), s_{t+1})            公理 3
```

| 项 | 值 |
|---|---|
| 优化器 | AdamW(lr 2e-4, wd 1e-4) |
| 调度 | OneCycleLR(pct_start .15, cos) |
| batch | 2048 |
| 步数 | **20k–40k**（V7 只用 3600 步——它只需复述被喂进去的答案；V8 要自己算几何，预算放大一个数量级） |
| BPTT | 截断，序列 128 帧 |
| 热启动 | **无**。schema 变了，V6/V7 权重不可复用（`load_policy` 会显式拒绝，这是对的） |
| 课程 | 先在 skills 语料上练 `L_aim`（84.7% 帧有安全射击线，对 multi 的 32%），再混入 multi |
| GPU | 1–2 张 |

**Stage 1 出口判据（达不到不进 Stage 2）**：
- `L_aim` 的 hit_enemy / hit_self 分类 **AUC ≥ 0.90**
- 离线 steer 误差中位数 **≤ 1 个锥宽**
- fire precision ≥ 0.85 且 recall ≥ 0.65（V7 达到 0.943 / 0.772，这是已证可达的水平）

理由：Stage 2 无法修复一个不会瞄准的策略，这是 V4–V6 已经付过的代价。

### Stage 2 — RL（**等用户验收 Stage 1 后才启动**）

- `--objective win`（`env.mjs:83-84` 给 `+(done && roundWinner === seat)`）
- 稠密塑形复用 V6 已实现的 `shaped_returns`：`kill_aux` 击杀增量 + `survival_bonus` + `death_penalty`。**这是补上 `frag = kills − suicides` 对"被第三方打死"完全沉默的那个洞**，per-n 数据（单挑 0.625 超教师、n≥3 崩到 0.08–0.15、n=6 为 0.000）证明这正是混战崩溃的原因
- **新增平局压力项**：随 roundTime 递增的负奖励。V7 greedy 平局率 24.3% 对教师 17.9%，超时=所有人 win=0，这一项此前所有版本都没有
- Critic 输入 `[z, g, pooled(m_1..m_7)]`
- 探索：只对 fire 的 Bernoulli 加温
- `max_kl` 超限回滚整步并重建优化器（V6 已有保护，保留）
- 课程 ramp：对手强弱取自 `arena/report-seed20260101.json`，n 从 2→8。**V5/V6B/V6C 唯一被实测有效的改动，保留**
- `teacher_aux = 0`（优势加权与均匀权重两种教师引导都被实测否证：KL 全部超限回滚）
- 128 worker

**Stage 2a — 先修自杀（2026-11-10 用户指定，实现 `policy/rl/v8/rl.py`）**

Stage 1 出口判据未达标，用户决定直接用现有基座进入 Stage 2，第一目标是把自毁压下来。

- **基座整体冻结**：trunk、drive/steer、基座 fire 头、辅助头都不更新。只训一个零初始化的
  安全旁路 `自身 8 + 自中心 lidar 64×3 + 时钟 1 → 64 → 1`，叠加到开火 logit。末层零初始化
  保证旁路为 0 时基座逐位不变（140 局 greedy 的 10 项指标与冻结评测完全相同）。
- **为什么需要旁路**：Stage 1 的 `aim_safe_recall=0.004`，说明 512 维 z 里没有线性可分的
  "这一炮会不会打回自己"；而前向墙距 + 法向正是近距离自击的物理原因。只调基座 fire 头
  的线性层（第一版）会让 p 与自击一起漂移。
- **逐发归因**：`emission`/`hit`/`self_hit`/`despawn` 都带 `shot_id`，奖励记到开火那一帧。
- **奖励**：击中敌人 +1，击中自己 −1，空放 0，hold 0，哑火 −0.05。
  - hold/空放为 0 不是逐字实现"不发射也 disadvantage"。实测逐帧 hold 惩罚 −0.05 会让
    "打一发空炮"相对继续 hold 变成正 advantage（且开火后的冷却帧不计 hold），而空放占
    85%、哑火占 21%，策略因此学会刷墙：fire 率 0.16→0.53、击杀→0。
  - 空放相对自击仍是 advantage（0 vs −1），保留用户要的方向；"不发射"不再是显式负奖励，
    因为基座 fire 头冻结后已经提供开火行为，旁路只需要学 veto。
- **算法**：one-step advantage `reward − V(s)`，V 是冻结 z 上的小 MLP，按当前决策分布拟合
  即时奖励；不做"只对开火行"的零均值归一化（那会抹掉"平均开火是亏的"这个信号，并把占
  多数的空放行变成正 advantage）。PPO clip 0.2，KL 早停 0.02。
- **修正的 bug**：`logp` 曾误存为 `binary_cross_entropy_with_logits`（−log π），导致
  `ratio = exp(new−old)` 实为 `exp(logπ_old−logπ_new)`，策略梯度符号反转。线性版、旁路版、
  两种奖励下完全一致的"p 越训越高、自击越训越多"都是这一个符号错误。
- **rollout**：96 Node worker、每 update 512 局、early-death（死亡且子弹结算完即终止）、
  每 8 个 update 用 70 局 development 评测；正式验收仍用冻结 `policy/rl/v8/evaluate.py`。

### Stage 3 — 导出与部署

图里无求解器 → 导出只剩 V2 当年已解决的常规工作（`arena/RL_LOCAL.md:37` 记录过
`atan2(0,0)` 的处理）。

- batch=1 静态、opset 17、GRU 单算子
- 校验三件套（本轮为 V7 建的那套，脚本在 `v7-runs/audit/`）：
  1. 静态帧对齐 — 真实帧 + 构造边界帧（全零 / alive-无候选 / dead），要求 drive/turn/fire/argmax 全等、输出有限
  2. 闭环引擎对齐 — 70 局真实对局，ONNX 驱动 vs PyTorch 驱动，要求逐局胜负序列一致
  3. 浏览器 WASM — sha256 校验 + `arena/test-rl-tracker.mjs` 观测一致性
- 接进 `arena/rl-controller.mjs` 作 `rl_v8`（接线模式本轮已为 V7 打通）

---

## 7. 验收（不再自定门）

| 项 | 值 |
|---|---|
| 集合 | `combat('development', dev_blocks, seed=20261107)` |
| 选点 | `dev_blocks=2` → 140 局 |
| **正式验收** | **`gate_blocks=32` → 2240 局/模式** |
| 聚合 | `policy.rl.v4.gates.mean()` — **n 分组宏平均**，不是行平均 |
| 区间 | `gates.interval()` 配对 bootstrap，4000 次重采样 |
| 胜率门 | `CI_lower(student_win − 0.7·teacher_win) > 0` |
| 自杀门 | `CI_upper(student_suicides − teacher_suicides) ≤ 0.03` |
| 解码 | `min(greedy, sampled)` 两者都要过；sampled 必须与训练同分布，**不许事后调温** |
| 参考值 | teacher hy 宏平均胜率 **0.301871**（本轮已两次独立复现），门槛 **0.211310**，自杀上限 **0.153044** |
| 随机基线 | 必须**实测**，不是解析 0.200。本轮实测均匀随机策略 = **0.0119**（140 局赢 1 局，自杀率 0.926） |

V7 的教训：`report.py:48` 用行平均、`:96` 把 `0.2` 硬编码，导致同一份权重报出
0.1143 而冻结口径是 0.1641（低估 43.6%）；且 140 局连 distill1 与 distill3 都分不
出来（配对 CI 跨 0）。

---

## 8. 唯一的风险点，与零成本对冲

**风险**：网络能否从「原始位置 + 墙格 + lidar」学出**多次反弹**的射击几何。

**为什么是真风险**：本轮消融证明 V7 的 8 维解析摘要冗余（抹掉后 13124 帧只改 1 个
决策），但那次消融**保留了 384 维瞄准块**——编码器当时在读现成答案。整块拿掉是不同
的问题，没有证据。

**架构层面的四条对冲（已写进设计，不是待办实验）**：
1. lidar 192 维给反射几何正确归纳偏置（距离 + 法向 = 反射定律的两个输入）
2. 地图走 CNN，卷积是学射线传播的自然基底
3. `L_aim` 在 64 方向固定基上稠密监督，每帧 192 个标签
4. 训练预算放大到 20k–40k 步（V7 的 6–11 倍）

**零成本对冲**：Stage 0 采集时**同时存下求解器输出作为标签块**。若 Stage 1 出口
判据不达标，把标签块的一部分翻回输入即可——不重采一帧、不改 schema。这个门免费留着，
只在失败时用。

预期：0 反弹与 1 反弹学得好，高反弹差一些。教师本身主要靠 0–1 反弹取胜，可承受。

---

## 9. 双教师 / 双 policy（已定）

最终要交付**两个 policy**：一个重胜率、一个重击杀。冻结框架里本来就是成对的
（`policy/rl/v4/common.py:10`：`TEACHERS = {'win': 'hy', 'frag': 'gpt'}`），而两位教师的
实测画像正好覆盖两个方向：

| 教师 | 胜率(宏) | 存活帧占比 | 自毁 | 击杀 | netKills |
|---|---|---|---|---|---|
| hy | **0.3019** | **0.840** | **0.121** | 0.850 | 0.729 |
| gpt | 0.243 | 0.693 | 0.229 | **1.157** | **0.929** |

gpt 强在击杀（V7 已修好的维度），hy 强在存活与胜率（V8 要修的维度）。所以**两个都采**。

| | 重胜率 policy | 重击杀 policy |
|---|---|---|
| 教师 / style 条件 | hy | gpt |
| Stage 2 目标 | `--objective win` | `--objective frag` |
| 验收门（冻结配置已定义） | 胜率 ≥ 0.7×0.3019 = **0.2113**；自杀 ≤ 0.1230+0.03 = **0.1530** | netKills ≥ 0.7×0.929 = **0.650**；自杀 ≤ 0.229+0.03 = **0.259** |
| 部署 | `rl_v8_win.onnx` | `rl_v8_frag.onnx` |

**Stage 1 只训一个模型，两个 policy 在 Stage 2 分叉。** 理由：Stage 1 九成的工作量是学
瞄准几何，而瞄准标签来自求解器、**与目标函数无关**，复制两份是浪费最贵的那部分。
风格区分靠已有的 `Encoder.style`（9 类 embedding），且赋的是**正确** style——V4 那次失败
是随机赋 style 制造标签噪声，本质不同。

成本为零：6000–10000 worlds × 2 席（hy + gpt）与 12000–20000 worlds × 1 席是同样的
帧数、同样的采集时间、同样的存储。

**Stage 1 出口判据新增一条**：`style=hy` 与 `style=gpt` 下的 drive 行为必须有可测差异
（存活帧占比或 drive 分布 KL），证明条件化真的生效。若无差异，退化为训两个独立
Stage 1 模型——代价只是多一次 GPU 训练，不影响数据。

---

## 10. 资源护栏（多人共用机器）

| 项 | 当前 | V8 上限 |
|---|---|---|
| 核 | 256，`yjh` 占约 60，load 15 | 采集 96 worker / RL 128 worker；load > 150 自动降 |
| 内存 | 247G，可用 207G | 采集常驻 < 20G；训练 < 30G |
| GPU | 8 张全空 | **≤ 2 张**（项目既定约束） |
| 磁盘 `/` | 439G 可用（88% 满） | 新语料 ~40G |
| 磁盘 `/data` | 5.1T 可用（98% 满） | **不写**（NFS 曾挂死） |

---

## 11. 与旧版的差异摘要

| | V6/V7 | V8 |
|---|---|---|
| 每帧输入 | 1966（含 800 维求解器输出、28 维 padding、24 维自身动作史） | **608**（求解器输出 0） |
| 回合级输入 | 混在每帧里，每帧重算 | **346**，缓存 |
| 世界记忆 | 无 | 7 个对手槽 GRU + 1 全局 GRU |
| 对手动作 | 不观测 | **21 维/帧** |
| 动作 | 102 类联合 | **5 维，3 个独立分布** |
| 转向 | 17 档离散 | 连续 tanh |
| 求解器角色 | 输入特征 | **仅训练期标签** |
| 辅助监督 | 1 项（aim 回归） | 4 项（aim / threat / intent / forward） |
| 训练步数 | 3600 | 20k–40k |
| ONNX | 导出失败（`adaptive_avg_pool2d` + `atan2(0,0)` + 回退链坏 + style 烤错） | 图中无求解器，无此四类问题 |
