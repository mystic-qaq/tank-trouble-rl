# V4 正式运行：2026-11-07

用户已授权迁移数据并重新启动。本轮仅使用 GPU 0–3，通过 UUID 限制可见设备，不启动旧 V1/V2/V3 训练。
启动后检查：迁移完成，20,842 个文件共 89,060,258,708 字节全部核对通过；新的 G0 已通过，共同预训练已实际更新到至少 800 步并使用 4 卡。G1 尚未验收，不能据此宣布初始化合格或进入 DAgger/PPO。
网络、奖励、G0–G4 验收阈值与 V4 方案相同；本轮改动是发射帧索引、成组随机读取、监督训练四卡梯度聚合，以及 G1 通过后两条目标分支的隔离并行调度。

## 存储

- 历史运行与示范：`$RL_ROOT/runs`。
- 兼容路径：`$RL_ROOT/runs`，迁移内容核验后改为上述目录的软链接。
- 当前运行：`$RL_ROOT/training/v4-20261107-main`。
- 临时文件与缓存：`$RL_ROOT/tmp`、`$RL_ROOT/cache`。
- 迁移进度与完整校验清单：`$RL_ROOT/operations/runs-migration.json`、`runs-migration-files.json`。

迁移先复制、逐文件验证内容并检查原件没有变化，再安装软链接，最后删除已验证的旧目录。不是仅核对文件大小后就删除原件。
新任务使用数据盘的规范路径，重新执行 G0；不修改旧证书的路径来伪装通过。

## 调度与验收

- 主会话：`tank-v4-main`。
- 固定源码：`$RL_ROOT/releases/20261107-v4-data4gpu`（只读）。
- 源码指纹：`4affccc13aa0cbf468a508e6b3f41a122a68b031900c7c584e72e18a0b00cc34`。
- 启动：北京时间 2026-11-07 02:05:57；总预算截止 10:05:57。上限不是成功条件。
- 日志：`$RL_ROOT/operations/v4-main.log`；启动参数：`operations/v4-launch.json`。
- G0 成功后，共同预训练最多 4 卡，保持全局 batch=512。
- 只有共同 G1 证书通过，才可启动两个目标分支。胜率分支隔离到物理卡 0、1，净击杀分支到 2、3，各自最多 32 个环境 worker。
- 每条分支仍独立执行 DAgger 验收、PPO 试运行验收和最终实战验收；一个分支失败不会放行该分支，也不会自动重试。
- CPU 回归 31 项通过、2 项因主动禁用 CUDA 而跳过；另行四卡测试中，单卡/四卡加权损失与梯度一致性及概率重算检查通过。

不能因为当前目录存在 `candidate.pt` 或损失下降，就宣布已通过 G1 或进入 PPO。

## 曲线与状态

TensorBoard：远端 `127.0.0.1:6008`，本机 http://127.0.0.1:16008/ ，会话 `tank-v4-board`。
本机 SSH 转发：

```powershell
ssh -N -L 127.0.0.1:16008:127.0.0.1:6008 gpu-host
```

```bash
cat $RL_ROOT/training/v4-20261107-main/pipeline.json
cat $RL_ROOT/training/v4-20261107-main/pretrain/status.json
tail -n 30 $RL_ROOT/operations/v4-main.log
```

后续分支的进度在 `win/branch.json`、`frag/branch.json`，通过证据在各阶段的 `acceptance.json`。
G0 审计期间 GPU 可以空闲，这时尚未更新模型。训练期间以状态和真实实战验收判断进展，不以显存是否占满判断成功。
