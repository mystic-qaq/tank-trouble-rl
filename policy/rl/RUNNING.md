> 当前已启动 V3 主线：约 462 万参数 GRU 联合动作模型，共享预训练 → 三轮分方向 DAgger → 奖励训练，最多使用 4 张 GPU。详见 [V3 运行说明](v3/RUNNING.md)，输出 `runs/v3`，曲线 http://127.0.0.1:16007/ 。
>
> V2 已完成有界初始化验收，四个候选均未通过全部前置门槛，V2 训练已停止且没有进入 PPO。详见 [V2 最终结果](v2/RESULTS_V2B.md)。本机观战页仍使用 V2 权重。
>
> 以下是旧 V1 的历史记录。V1 已按用户要求停止并继续保持 `stopped_by_user`；其训练、评估和监控不会恢复。
> 胜率任务最后记录 782,336 帧；净击杀任务最后记录 892,928 帧。检查点、教师数据、日志全部保留。
> 服务器状态已标记 stopped_by_user。重新设计前不恢复本轮或启动新训练。

# Server3090 首轮运行记录

正式游戏、Arena 和训练统一为 60 秒，超时平局。2026-11-06 已启动首轮实验。

- SSH 别名：gpu-host。
- 工作目录：$RL_ROOT。
- 固定源码快照：$RL_ROOT/releases/20261106-60s-v1。
- 胜率实验：GPU 0，tmux 会话 tank-rl-win60，输出 runs/initial60/win。
- 净击杀实验：GPU 1，tmux 会话 tank-rl-frag60，输出 runs/initial60/frag。
- 每个模型已采集 65,536 条教师样本，完成 30 个 BC epoch，以及教师/学生各 28 局开发验证。
- 两条 PPO 实验已开始，每条预算 1,048,576 个环境帧，每 262,144 帧做开发验证并保存候选。
- 初始 BC 学生尚明显弱于教师；本轮用于检查学习曲线，不代表已经达成替换条件。

最近检查（Asia/Shanghai）：2026-11-06T17:18:41.079986+08:00。

- win: running, ppo, 28,672 帧，413.5 环境帧/秒。
- frag: running, ppo, 28,672 帧，393.9 环境帧/秒。

进度检查：在 gpu-host 上运行 `python3 $RL_ROOT/policy/rl/status.py`。
日志为 runs/initial60/win.log 与 frag.log。SSH 断开不终止 tmux 内的任务。
恢复使用根目录 policy/rl/train_ppo.py 的 --resume，命令见 README.md；已验证 CUDA RNG 检查点恢复。
正式 Bot/minimax 尚未替换。旧七份策略的冻结校验保持通过。

已验证：物理回归、2–8 人精确 3600 帧超时、重复策略槽位、逐帧重放、死亡后真实子弹击杀、
自毁/获胜奖励、截止帧胜利优先于超时、模型概率与 padding mask、GAE 终止/截断、CUDA 恢复、
教师采集/BC/PPO/开发实战全流程，以及正式游戏与 Arena 的浏览器 60 秒平局和胜场不变。
