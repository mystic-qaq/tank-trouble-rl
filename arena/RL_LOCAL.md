# 本机观看 V2 四方对战

在项目根目录执行：

```powershell
node arena/serve.mjs --port 8765
```

然后打开 http://127.0.0.1:8765/arena/?preset=v2 。
模型、推理运行库和游戏均在本机；观战不需要 SSH、Python 或 Server3090，也不会启动训练。
默认 0.5 倍速，可以暂停、单步或调整速度。实际速度低于设定值时会显示实际倍率，物理步长仍固定为 1/60 秒。

## 默认四名参赛者

| 名称 | 颜色 | 来源 |
|---|---|---|
| RL 胜率 | 紫色 | V2b `win-fw4/bc/best.pt`，DAgger 回训 step 7,500 |
| RL 净击杀 | 绿色 | V2b `frag-fw4/bc/best.pt`，DAgger 回训 step 500 |
| minimax | 橙色 | 原冻结策略 |
| 内置 Bot | 粉色 | 原引擎 Bot |

这是自由混战，不是组队比赛。每回合至多 60 秒，超时平局；每 4 局轮换全部席位，再换地图。
排行榜显示累计胜场、击杀、自毁与真实发射数。重开本局会撤回该局已计入的统计，避免重复计分。
也可以取消勾选参赛者，选择 2–8 个 AI 做单挑、三方或其他组合对战；“V2 四方对战”按钮可恢复默认阵容。

这两个模型仅完成 BC＋一次 DAgger，PPO 未启动，尚未通过训练计划的全部验收。
胜率候选 B 的技能更稳定，净击杀候选 B 的教师开火召回较高，因此先选择这两份作观察。
单个布局、单个回合不能用于判断整体强弱。

## 接入与校验

- 模型原始权重保存在 `policy/rl/v2/checkpoints/`，SHA-256 与服务器保留检查点一致。
- `arena/models/v2-*.json` 记录源权重与 ONNX 哈希、候选和训练步数。浏览器加载时校验 ONNX 哈希。
- ONNX actor 使用与 Python 相同的确定性动作：移动/转向类别取 argmax，转向幅度 sigmoid，条件开火阈值 0；不增加偏置。
- 沿用训练的 1,966 维观测、动作历史与 60Hz 决策。两个模型都在整帧开始时观察世界，不在其他坦克已移动后重新观察。
- 每个模型的 258 个输入（包含空实体/填充边界）分别与 PyTorch、原生 ONNX Runtime 和浏览器 WASM 比较。浏览器移动方向与开火决定全部一致，连续控制最大误差低于 1e-6。
- 导出时修正了 legacy exporter 对 `atan2(0,0)` 的非有限值问题，使其保持 PyTorch 的 0；未改动网络权重。
- 额外校验了 480 帧浏览器观测追踪器与训练环境逐值一致，600 帧固定种子重放一致，以及完整回合、暂停与固定单步。
- 原七份固定策略仍通过 7 个策略/74 个文件的冻结校验。

本机检查命令：

```powershell
node --experimental-vm-modules --disable-warning=ExperimentalWarning arena/test-rl-tracker.mjs
node arena/regression.mjs
node policy/verify-frozen.mjs
```

原始浏览器检查结果位于 `output/playwright/v2-browser-parity.txt` 和 `output/playwright/v2-match-check.txt`。
ONNX Runtime Web 1.31.1 以本地静态文件提供，许可证见 `arena/vendor/onnxruntime/LICENSE`。

原版八个策略的观战入口仍是 `/arena/`；没有选择 V2 时不会加载神经网络。原版页面仍可直接打开 HTML，V2 模型需使用上述本机 HTTP 服务。
