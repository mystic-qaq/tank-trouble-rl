# V4 Policy 训练诊断与修改建议（2026-11-07）

> 本文件只记录诊断与建议，不含任何代码改动。所有结论基于 gpu-host 上
> `$RL_ROOT/training/v4-20261107-main` 的最终产物与本地源码快照
> `releases/20261107-v4-data4gpu` 的只读检查。

## 0. 结论速览

三个相互独立的问题，必须分开处理：

| # | 问题 | 性质 | 后果 |
|---|---|---|---|
| A | 开火类被帧级重要性权重压掉 199~1000 倍 | 训练信号设计缺陷 | G1 必然失败，加步数/加卡都无效 |
| B | 训练与部署的 logits 温度口径不一致 | 隐性实现缺陷 | PPO 的 BC 辅助项与 rollout 优化的不是同一分布 |
| C | 4 张 3090 长期被占但利用率 8~11% | IO/并行策略错配 | 8 小时预算里算力几乎没用上，且 NFS 故障时必然挂死 |

优先级：**先修训练信号，再修 IO/GPU**。只修 C 会让模型更快地再次失败。

## 1. 事实基线

### 1.1 最终状态（04:29:39）
- `pipeline.json`: `stopped_not_accepted`，`stage=G1_pretrain`，`error=GateFailure('Pretraining development/offline criteria not met; DAgger is forbidden')`，`next_stage_allowed=false`，`automatic_restart=false`。traceback 指向 `train.py:101`。
- `pretrain/status.json`: `step=30000 / 40000`，`loss=0.7671`，`gpus=4`，`training_not_accepted`。
- 停止是 **patience 早停**：最佳检查点在 step 15000（03:19 写入 `candidate.pt`），之后 20000/25000/30000 连续三次无改善 → `stale >= patience(3)` 退出。预算 28800s 只用掉约 2h24m，与 10:05 的截止、06:00 的 NFS 故障都无关。
- `win/`、`frag/` 目录从未创建，没有任何 `acceptance.json`。两个目标 policy 从未开始训练。

### 1.2 六次开发评估的最好值 vs 门槛
| 指标 | G1 门槛 | 六次最好值 | 备注 |
|---|---:|---:|---|
| win 移动准确率 | 0.85 | 0.433 | 单调爬升但远不及格 |
| frag 移动准确率 | 0.85 | 0.545 | 同上 |
| win 开火精确率 / 召回 | 0.70 / 0.60 | **0.000 / 0.000** | 85,225 样本、572 正例，六次 `fire_predictions` **恒为 0** |
| frag 开火精确率 / 召回 | 0.70 / 0.60 | 0.616 / 0.136 | 最多只预测 271 次 |
| win 胜率（greedy/sampled） | 0.211（70% 教师） | 0.081 / 0.058 | 教师 0.302 |
| frag 净击杀（greedy/sampled） | 0.591 | 0.196 / 0.219 | 教师 0.845 |

`bc/loss` 从 1.79 降到 0.767，但能力指标几乎不动 —— loss 下降主要来自学会了"永远不开火 + 常数动作"这个低熵解。

### 1.3 数据分布（G0 收据实测）
| 项 | 值 |
|---|---|
| 轨迹文件 / 数据量 | 10,519 个 .npy / **81.4 GB** |
| 总帧数 / 发射帧 | 10,334,998 / 62,063 |
| **自然开火率** | **0.601%** |
| 单轨迹长度 | 中位 792 帧，最大 3600；发射帧中位 4，979 条轨迹为 0 |
| hy 训练 / 发射帧 | 1,707 条 / 17,785 |
| gpt 训练 / 发射帧 | 1,619 条 / 15,969 |
| 离线验证 | hy 174 条→85,225 样本/572 正例；gpt 205 条→94,405 样本/1,226 正例 |
| 教师风格 | 9 个 style 共享一个模型，hy/gpt 各约占训练轨迹 16% |

### 1.4 资源实况
| 项 | 值 |
|---|---|
| 硬件 | 8×RTX 3090 24GB；247 GiB RAM，**206 GiB available**；根盘 523G 可用 |
| 启动分配 | `CUDA_VISIBLE_DEVICES` 显式 4 张 UUID，`gpu_limit=4`，`--device cuda:0` |
| 稳态 GPU | 4 卡各 **550~644 MiB / 24576 MiB（2.3%）**，利用率 **8~11%**；另 4 卡完全空闲 |
| 进程 | 单进程 `Rl+`、CPU 104%、RSS 4.4GB、`rchar 440.9 GB`、`syscr 621 万次`、`wchan=folio_wait_bit_common` |
| 步速 | 训练段约 **10 步/秒**；每 5000 步周期实测约 **22 分钟** |
| 模型 | 3,582,086 参数 / **14.33 MB** / 102 类联合动作头 |

## 2. 根因 A：帧级重要性权重把开火类从目标函数里抹掉

`data.py:169-174`：

```python
proposal = .5 / length + .5 / len(positives) * (fire_label)
importance.extend((1 / length) / proposal)
```

`Frames.batch` 先做 50% 开火帧过采样（第 166-167 行），再用精确逆概率把分布**校正回轨迹内均匀分布**。代入真实数据：

| 轨迹 | 非开火样本权重 | 开火样本权重 | 压制倍数 |
|---|---:|---:|---:|
| 中位 L=792, P=4 | 2.000 | 0.01005 | **199×** |
| L=3000, P=3 | 2.000 | 0.00200 | **1000×** |

于是每个 batch 里虽然 50% 是开火帧，但它们对梯度的有效贡献只有约 **0.6%**（正好等于自然开火率），其余 99.4% 都在教模型"不开火"。叠加三个放大器：

1. **102 类联合头**（`model.py:17,71`）：`3(drive) × 17(turn) × 2(fire) = 102`。开火不是一个二分类，而是 51 个 `(drive,turn,fire=1)` 组合；正例被摊薄到 51 个 logit 上，每个具体开火类几乎见不到梯度。
2. **风格稀释**（`data.py:130-138,172`）：9 个 style 共享权重，hy/gpt 各约 16%；且 `teacher == 'reference'` 的轨迹会把同一条状态随机标成 `hy/gpt/reference` 三种 style，制造同状态冲突标签。评估时 style 固定为单一教师，训练时却不是。
3. **损失未归一化**（`train.py:65`）：`(loss * importance).mean()` 除以样本数而非权重和，有效学习率随轨迹长度漂移。

结果就是 win 方向六次评估 `fire_predictions = 0`；frag 方向因为 gpt 的发射帧更多、模式更容易，才勉强学到 0.136 的召回。

**为什么这个 bug 没被 G0 抓住**：`test_data.py:26-32` 把"加权后开火率必须等于自然频率"写成了正确性契约：

```python
assert abs(float((actions[:, 2] * weights).mean()) - .05) < .01
```

而测试 fixture 是 100 帧 / 5 个开火帧 = **5% 开火率**，比真实数据 0.601% 高 8 倍。在这个 toy 分布下权重压制只有 ~21×，看起来无害；真实分布下是 199~1000×。「统计无偏」和「分类可学习」被混为一谈：过采样本来就是为了平衡类别，再乘精确逆概率等于把平衡完全撤销，回到极端不平衡，且梯度方差更差。

> 注意不要矫枉过正：V3 曾因开火过多导致随机采样 64 局自杀 55/51 次、击杀仅 4/5、KL 0.065/0.019 而停止。目标是「在正确时机开火」，不是「多开火」。

## 3. 根因 B：训练与部署的 logits 温度口径不一致

- `model.py:74-78`：`logits(..., training_distribution=False)` → `scores / temperature`（T=0.35，放大 2.86 倍）；`training_distribution=True` → 原始 `scores`。
- `model.py:105-108`：`Policy.forward`（BC 损失）走 **raw logits**。
- `rollout.py:79` / `ppo.py:25,47`：`act()`、PPO 新旧策略 logprob、KL 检查全部走 **带温度 logits**。
- `ppo.py:64` 的 BC 辅助项 `actor.bc_loss(...)` 又走 **raw logits**。

即：同一个 actor 在一个 update 里同时对两套不同温度的分布求梯度。argmax 不受单调缩放影响，但 softmax 概率、logprob、KL、采样熵全都受影响；BC 辅助项会系统性地把 logits 往大推，部署时分布更尖锐。这很可能放大了 V3 的 KL 越界与自杀率问题。`test_model.py` 目前**没有**覆盖这个一致性。

## 4. 根因 C：门槛本身可能不可达

移动准确率只有 0.43/0.55，而门槛是 0.85。在 60 Hz 单帧观测、9 个风格迥异的教师、随机化动作的条件下，教师下一步动作本身可能高度不可预测。建议**在改训练之前**先测上界：用验证集训练一个 teacher-conditional oracle、或测同风格教师自洽率/多轨迹同状态动作一致性。若上界 < 0.85，则门槛定义本身不可实现，应改为「按局势加权的关键帧准确率」或「动作分布 KL」，而不是靠加步数硬撞。这属于指标定义变更，需要显式记录并重新过契约测试，不能悄悄降阈值。

## 5. 根因 D：为什么 4 张卡长期占着不用

| 层 | 机制 | 证据 |
|---|---|---|
| D1 多卡对小模型负收益 | `train.py:54-58` 用 `DataParallel` 把全局 batch 512 切到 4 卡 → **每卡 128 样本**；模型仅 14.33 MB。DP 每步要广播参数、scatter 输入、gather 输出、Python 层同步，开销大于 128 样本的前反向计算。README:122-123 与 `test_model.py:52-74` 明确把「四卡一致性」写进了设计与契约 —— 这是 `data4gpu` 版本为「用满 4 卡」做的改动，但方向错了 | 4 卡各 550~644 MiB 是 CUDA context + 复制的模型，不是计算需要 |
| D2 每步被 NFS 随机小读阻塞 | `data.py:156` `groups = min(32, 512//16) = 32` → **每步从 10,519 个 .npy 随机打开 32 个文件**、各取 16 行；`cache` 只有 64 条，命中率极低；NFS `rsize=wsize=1MB, hard, timeo=600`，随机小读被放大成整块 RPC。`torch.set_num_threads(1)` + `OMP/MKL/OPENBLAS=1` 让取数全在主线程串行，`.to(device)` 同步拷贝，无 prefetch/pin_memory/non_blocking | `rchar 440.9 GB`、`syscr 621 万次`、CPU 104%、`folio_wait_bit_common`、10 步/秒 |
| D3 缓存与临时目录全在 NFS | 启动 env 把 `TMPDIR`、`CUDA_CACHE_PATH`、`TORCH_HOME`、`XDG_CACHE_HOME` 全指向 `/data`；而机器有 **206 GiB 可用内存**，数据只有 81.4 GB，完全可整份驻留 | `v4-launch.json` env；`free -h` |
| D4 评估占了约 60% 墙钟 | 每 5000 步一次 `dev`（2 objective × 2 mode，32 个 Node 物理仿真 worker）+ `offline`（379 个验证文件各 mmap 随机读 512 帧、512-batch 推理）。实测六个周期各 21~25 分钟，按 10 步/秒折算纯训练仅约 8 分钟/周期 | dev-*.json 时间戳：02:35 / 02:57 / 03:19 / 03:41 / 04:04 / 04:29 |
| D5 计算密度低 | fp32 且 `model.precision()` 显式禁用 TF32；无 AMP；batch 512 对 3.6M 参数模型偏小；单帧前馈网络没有大核算 | `model.py:21-24` |
| D6 结构性闲置 | 设计上 G1 占 4 卡，另 4 卡只在 G1 通过后由 `parallel_branches` 分成两卡组使用。G1 没过 → 8 卡中 4 张低效占用、4 张从未使用 | `pipeline.py:48-49,63-95`；`v4-launch.json` 的 4 个 UUID |

一句话：**瓶颈在「每步 32 次 NFS 随机读」和「每 5000 步一次 CPU 仿真评估」，不在 GPU 算力；加卡只会让每卡分到的 batch 更小、同步开销更大。** V3 的 `HEALTH_20261107.md:8` 也记录「始终绑定 GPU 0–3」，说明这是跨版本的结构性问题，不是 V4 偶发。

## 6. 修改建议

### P0 — 修训练信号（不做完这些，重跑必然再失败）

1. **取消帧级逆概率校正，改为归一化类别平衡权重**（`data.py:169-176`）。
   保留过采样但把比例从 50% 降到 **10~20%**，帧级权重直接取常数（或温和 `fire_w ∈ [0.25, 1]`，用验证集定）；损失改成加权均值 `(loss*w).sum() / w.sum()`（`train.py:65`），保证有效学习率不随轨迹长度漂移。无偏性由 `offline()` 的均匀验证采样保证，不需要也不应该由训练批次承担。
2. **同步改契约测试**（`test_data.py:26-32`）。把「加权后开火率 = 0.05」改成三条：
   - batch 中开火占比落在目标区间（如 10~20%）；
   - 归一化后权重均值为 1、权重范围有上界；
   - **开火类有效梯度贡献 ≥ 20%**（防止再次退化到 0.6%）。
   并把 fixture 的开火率改成真实量级（0.6%）或参数化多档，避免 toy 分布掩盖问题。新增一条「部署开火率相对教师 + 自杀率」的回归测试，防止回到 V3 的乱开火。
3. **拆分动作头**（`model.py:17,71,105`）：改为 `motion_head(51) + fire_head(2)`，fire 用 `BCEWithLogitsLoss`（可加温和 `pos_weight`），或保留 102 类联合头但对 fire 边缘（每对相邻 logit 的 logsumexp 差）加辅助 BCE。让「是否开火」不再与 51 个运动组合竞争。
   ⚠️ 这会改变 `ARCH`，必须整体新发快照、重跑契约；V3 权重本来就不可复用，成本可控。
4. **统一温度口径**（`model.py:74-78,105-108`、`ppo.py:64`）：BC 与 PPO 辅助项、rollout、KL 全部使用同一组 logits；温度只作为采样时的独立参数，不进入损失。在 `test_model.py` 增加断言：`bc_loss` 使用的 logits 与 `act()` 的 logits 一致（或仅差一个显式、可验证的常数缩放）。
5. **提高目标风格的有效样本量**（`data.py:130-138,172`）：G1 对 hy/gpt 上采样，其余风格作正则；`reference` 轨迹固定单一 style，不要对同一状态随机赋三种冲突标签。
6. **先测门槛可达上界**（见第 4 节），再决定是否调整指标定义。
7. **评估分级**：轻量 screening（少量局 + 常驻内存的离线验证张量）每 1000~2000 步一次用于 patience 判定；正式 G1 验收只在 screening 通过后跑一次（`train.py:77-99`）。避免每 5000 步花 14 分钟在无望指标上。

### P1 — 修 GPU 与 IO

8. **数据本地化（收益最大）**：81.4 GB < 206 GiB available。G0 审计之后，把数据集一次性**顺序**读入内存/本地盘（`/home`，根盘 523G 可用），`Frames.array` 优先命中本地副本，用收据里的 sha256 校验一次后训练期不再触碰 NFS。启动 env 的 `TMPDIR`/`CUDA_CACHE_PATH`/`TORCH_HOME`/`XDG_CACHE_HOME` 全部改指本地盘。
   ⚠️ 审计契约绑定 `/data` 原始路径与 mtime（`verify_data`），建议加一层只读 `LocalMirror` 缓存，**不要改收据里的路径**；或在新轮次直接以本地副本作为 `--data` 重新生成收据。
9. **打包小文件**：把 10,519 个 .npy 合并成按 teacher 分片的大 memmap + 偏移索引，把「每步 32 次元数据 RPC」变成一次顺序读；若必须保留逐文件审计，保留哈希清单、训练只读打包文件。
10. **单卡训练**：`fit()` 的 DataParallel 改为可配置，G1 默认单卡 `cuda:0`，`CUDA_VISIBLE_DEVICES` 只暴露 1 张；全局 batch 提到 **2048~4096** 提高计算密度；`plan()` 的 `bc_gpu_limit` 改 1，`maximum_authorized_gpus=4` 留给 G2 两分支。同步修改 `test_model.py:52-74` 的四卡契约（保留为可选 DDP 一致性测试）。
11. **异步取数**：`groups` 从 32 降到 4~8（每条轨迹多抽行），或按轨迹批量预取；后台线程 prefetch 4~8 个 batch，`pin_memory=True` + `non_blocking=True`；`torch.set_num_threads(4~8)` 给数据准备。
12. **评估加速**：把离线验证集（约 1.4 GB）一次性物化到内存并用 2048 大 batch 推理；Node worker 与 GPU 推理用队列解耦，避免 CPU/GPU 互等。
13. **计算密度**：确认确定性要求后评估开启 TF32 / AMP（会改变数值与契约，需重新做 G0 一致性检查）；至少去掉 `precision()` 里对 TF32 的硬性禁用并实测精度影响。
14. **让低效可见**：在 `status.json` 与 TensorBoard 增加 `data_ms`、`compute_ms`、`eval_ms`、`steps_per_sec`、`gpu_util`、`nfs_read_rate`，以及 `bc/fire_predictions_on_validation`、`bc/effective_fire_weight`。设一条硬规则：连续 N 分钟 GPU < 30% 且步速低于阈值即告警/早停，避免 8 小时预算空烧。

### P2 — 可靠性

15. NFS `hard,timeo=600` 对长训练是致命的：06:00 那次服务端 nfsd 挂死让 60 个进程 D 状态数小时。训练数据本地化后即天然规避；否则至少 `soft` + 快速失败写 `failure.json`。
16. **日志写本地盘**：本次 `v4-main.log` 直到 traceback 前一直 0 字节（stdout 重定向到 NFS 文件），`RUNNING.md` 让人 `tail` 它拿不到任何东西。改为写 `/home` 再异步同步，且每个评估点 flush。
17. TensorBoard 的 `--logdir` 目前只指向旧运行目录；新轮次要用新目录，需把 TB 指到父目录或重启，否则看不到新曲线。

## 7. 改动后的强制流程（不能跳过）

- `common.py:34-39` 的 `source_digest()` 会散列 `js/`、`arena/`、`policy/` 下**所有** `.py/.mjs/.json`。任何代码改动都会改变指纹 → `contracts.json` 与 `data-acceptance.json` 立即失效 → `verify_data` 会拒绝。
- 因此必须：新快照目录 → 新 `--out` 目录（`exist_ok=False`，不能沿用 `v4-20261107-main`）→ 重跑 `preflight`（目标：**31 passed, 2 skipped**）→ 重新逐文件审计生成 G0 收据（20,842 文件 / 81.4 GB / sha256 + mtime + fire_rows）→ 再开始 G1。
- 建议先用 `--prepare-only` + 2000 步 smoke 验证「开火类确实学到东西」（验证集 `fire_predictions > 0`、有效权重 ≥20%、GPU 单卡利用率 >60%、步速 >50 步/秒），通过后再给完整预算。
- 不允许降低 G1 阈值、跳过 G1、从 `candidate.pt` 续跑或用 V3 权重初始化。

## 8. 本次检查的边界

只读检查：未改动服务器或本地任何训练代码、数据、配置、进程；未重启训练、未清理文件、未 umount。本文件是新增的 `.md`，不参与 `source_digest()`，不影响任何契约。
