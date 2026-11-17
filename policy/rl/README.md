# Tank Trouble RL · 第一轮实验

> **历史文档（2026-11-07）。** 这里记录的是 V1–V4 时期的运行状态，不是项目的最终结论。
> 最终结果、失败复盘与复现说明见根目录 `README.md` 与 `docs/03-learning-policies.md`。


> V3 已停止。用户授权后已启动 [V4 正式运行](v4/RUNNING.md)，数据迁至 `$RL_ROOT`，最多 4 卡；各阶段仍按[完整方案](v4/README.md)独立验收。下文保留旧 V1/V2 历史说明。

> V2 已完成初始化验收，四个候选按门槛停止，未启动 PPO。请查看 [最终结果](v2/RESULTS_V2B.md)、[V2 运行说明](v2/RUNNING.md) 和 `policy/rl/v2/status.py`。V1 保持停止。
> 下文旧训练命令仅作为 V1 记录保留；当前 V2 的数据、模型、动作分布和验收门槛均已独立实现，不使用旧 PPO 命令恢复。

## 实时学习曲线

TensorBoard 服务运行在 gpu-host 的 `127.0.0.1:6006`。
本机通过 SSH 转发后打开 http://127.0.0.1:16006/ 。当前已为浏览器启用每 5 秒自动刷新。

```bash
# 本机终端；已有转发运行时不用重复执行
ssh -N -L 127.0.0.1:16006:127.0.0.1:6006 gpu-host
```

服务器上的 `tank-rl-events` 会话读取既有日志并持续写 TensorBoard 事件，`tank-rl-tensorboard` 会话提供页面。
它们与训练进程分开，不修改正在运行的源码快照、模型、随机状态或采样设置。
监控服务重启：`bash policy/rl/start_monitoring.sh`，已有会话会保留。

- `win` / `frag` 是两个学习模型；`HY_baseline` / `GPT_baseline` 是固定教师基线。
- `Training/mixed_recent_win_rate` 和 `mixed_recent_net_kills` 是探索时最近 100 局的原始混合人数成绩，不能直接与固定评估基线比较。
- `Evaluation/win_rate` 和 `net_kills` 是固定开发场景的确定性实战成绩；每 262,144 帧评估一次，与教师使用同一批场景。
- `Training_by_N` / `Evaluation_by_N` 分人数显示，避免人数混合造成误判；监控首次接入时，分人数训练曲线从接入时刻开始，整体训练历史已回填。
- `BC` 横轴为 epoch；`Training` / `Evaluation` / `Optimizer` 横轴为环境帧。训练曲线可适度平滑，但应同时查看原始值。
- 评估期间采样与优化会暂停，因此训练图上数分钟没有新点可能是正在评估。开发集仅 28 局，不能用它宣称已经超过 HY/GPT。

当前网络共 159,047 个参数，FP32 参数约 0.61 MiB，整张卡约 426 MiB 使用量包含 CUDA 上下文、缓存、优化器和 rollout。
独立短测：16 环境一步的模拟/通信 28.6 ms、模型推理及动作回传 2.8 ms；模拟/通信占采样探针约 91%。
512 样本的一次探针优化约 11.1 ms。GPU 低负载主要反映当前小模型、同步小批量采样与 CPU 引擎供给限制。
这些是当前服务器负载下的短测，不是精确的生产阶段占比；优化应看环境帧/秒和实战成绩，而不是单独追求显存占满。

正式游戏、Arena、训练统一使用 60 秒交战上限，超时记平局，胜场不增加，已有净击杀保留。
共用配置是 `js/util.js` 中的 `ROUND_TIME_LIMIT`，计时只包含 playing，不含开场/结算。
旧 90 秒战报不能直接用来判定这两个新模型是否达标。

## 实现范围

- Node 原始引擎逐帧交互，60 Hz；2–8 人、重复策略对手、随机地图/出生/人数/被控席位。
- 两个目标独立：win 为最终获胜 +1，其余 0；frag 为击杀他人 − 自毁。
- 评估始终跑完整回合。训练中胜率模型死亡即可判负；净击杀模型死亡后继续结算，直至己方子弹全部消失或回合结束。仅在最终收益确定后提前终止，死亡阶段屏蔽 actor loss。
- Python/PyTorch 批量小网络；926 维输入、实体有效性 mask，三个动作头包含连续转向。
- 教师采集、按完整回合划分的 BC 数据验证、模仿初始化、PPO、开发集实战验证与检查点。
- PPO γ=1，GAE λ=.98；正确区分 rollout 边界和回合终止，终止后自动 reset。
- 训练场景 seed 限制在低 30 位；开发验证 seed 使用 0x40000000 起的独立区间。
- 每个对手槽位单独的随机流；逐局新沙箱，逐帧确定性重放。

这是一轮训练管线验证，不代表模型已经超过 HY/GPT。首轮仍使用原始冻结对手池，
历史模型联赛、跨名单正式留出测试、大规模配对簇置信区间、浏览器模型导出与替换尚待后续实现。
28 局开发验证只用于检查和挑选候选，不能作为替换旧策略的充分依据。

## Server3090

项目根目录：`$RL_ROOT`；Python 环境：该目录的 `.venv`。
原有七份策略保持冻结。初始实验分别使用物理 GPU 0 和 1，各自 16 个环境；不占满全部八张卡。
正式启动的实验使用 `releases/20261106-60s-v1` 内的源码快照，防止后续编辑混入正在运行的训练；
输出仍统一写到根目录 `runs/initial60`，每个实验保存源码 SHA-256 清单。

安装（已有环境无需重复）：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r policy/rl/requirements.txt
```

回归：

```bash
node policy/verify-frozen.mjs
node arena/regression.mjs
node --experimental-vm-modules --disable-warning=ExperimentalWarning policy/rl/test-env.mjs
.venv/bin/python -m pytest -q policy/rl/test_model.py
```

完整首轮（命令需在根目录执行，建议在独立 tmux 会话中运行）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u policy/rl/run_experiment.py \
  --objective win --out runs/initial60/win --samples 65536 --steps 1048576 --envs 16

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u policy/rl/run_experiment.py \
  --objective frag --out runs/initial60/frag --samples 65536 --steps 1048576 --envs 16 --seed 20261107
```

每个实验按顺序执行：教师样本 → 30 个 BC epoch → 教师和学生各 28 局同场景开发验证 → 约 105 万环境帧 PPO。
教师数据先用 65,536 条检验可学习性；通过后再扩展到 20–100 万条，不承诺这批小数据已经足够复制教师实力。
PPO 每 262,144 帧做开发验证。`latest.pt` 每轮原子保存，`best.pt` 按开发指标保存，完成后写 `final.pt`。

状态文件：

```text
runs/initial60/{win,frag}/experiment.json
runs/initial60/{win,frag}/demonstrations/status.json
runs/initial60/{win,frag}/bc/status.json
runs/initial60/{win,frag}/ppo/status.json
runs/initial60/{win,frag}/ppo/episodes.jsonl
runs/initial60/{win,frag}/validation-teacher.json
runs/initial60/{win,frag}/validation-bc.json
```

tmux 能在 SSH 断开后继续运行。检查点恢复会继续模型与优化器、随机状态，环境重新从新回合开始，
不声称与完全未中断的运行逐帧一致。PPO 示例：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u policy/rl/train_ppo.py \
  --objective win --resume runs/initial60/win/ppo/latest.pt \
  --out runs/initial60/win/ppo --steps 1048576 --envs 16
```

查看进度（不依赖训练环境）：

```bash
python3 policy/rl/status.py
tmux list-sessions
tail -n 5 runs/initial60/win.log
tail -n 5 runs/initial60/frag.log
```

如果教师采集半途失败并留下没有 manifest 的数据目录，先保留该目录，再选择新的输出目录重采；
采集器会拒绝覆盖已有数据分片。BC 和 PPO 都保留现有模型，完整流水线可跳过已经完成的数据/初始化。

## 指标与边界

`frames` 是整个游戏世界推进一次 1/60 秒，不按坦克数量乘倍；`samples` 为存活阶段的教师状态动作样本。
BC 使用完整回合 seed 划分训练/验证；开发实战按人数等权汇总，报告胜率、净击杀/局与超时率。
初版开发场景尚未覆盖每种人数的完整席位轮转块，正式验收仍按 PLAN.md 扩展。
环境逐个回合重新隔离策略状态；通信进程与 worker 跨帧常驻。

当前正式 Bot/minimax 尚未替换。任何权重发布需先经过独立实战验收和浏览器推理一致性检查。
