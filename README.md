# Countercurrent CNN Proof of Concept

本仓库实现两份最新版研究规范中的第一阶段 CNN 实验：用 source evidence 与模型自己生成的
target hypothesis 作为两端 boundary，在推理阶段进行 counterflow paired exchange。

- [核心思想](COUNTERCURRENT_CORE_IDEA.md)
- [实验规范](COUNTERCURRENT_CNN_POC_EXPERIMENT.md)

## 新版主模型

第 0 轮是普通前向 CNN：

```text
x -> Stem -> H0 -> F0 -> H1 -> ... -> FL-1 -> HL -> logits(0)
```

随后每个 outer iteration `t=1..T` 都重新注入原图，并由上一轮预测构造右边界：

```text
p(t-1) = softmax(logits(t-1))
z_hyp  = p(t-1) @ class_prototypes
CL     = boundary_projector(z_hyp)

H0  ---> H1 ---> H2 ---> H3 ---> HL
         ↕       ↕       ↕       ↕
C0  <--- C1 <--- C2 <--- C3 <--- CL
```

每个位置先分别 transport，再以逐通道正 conductance 交换：

```text
D     = h_bar - c_ref
gamma = 0.49 * sigmoid(logit_gamma)   # 0 < gamma < 0.5
Q     = gamma * D
H_new = h_bar - Q
C_new = c_ref + Q
```

主模型不接收 label/GT，GT 只进入最终 Cross-Entropy；最终分类头只读取 `H_L`。这里的
`-Q/+Q` 只是 exchange substep 的 zero-sum 约束，不代表整个网络能量守恒。

核心实现位于：

- `countercurrent_nn/models/coupled.py`：two-sweep closed-loop inference；
- `countercurrent_nn/models/boundary.py`：soft prototype target boundary；
- `countercurrent_nn/models/exchange.py`：channel-wise conductance 与 paired exchange；
- `countercurrent_nn/models/countercurrent.py`、`cocurrent.py`：主模型和同向对照；
- `countercurrent_nn/models/recurrent_forward.py`：equal-compute forward recurrent 对照；
- `countercurrent_nn/models/feedback_fusion.py`：普通 feedback fusion 对照。

## 实验组

| 配置 | 目的 |
|---|---|
| `cifar10_single.yaml` | 普通 feed-forward CNN |
| `cifar10_single_parammatched.yaml` | 参数量近似匹配的更深单流 CNN |
| `cifar10_forward_recurrent.yaml` | 无 reverse 的迭代计算对照 |
| `cifar10_feedback_fusion.yaml` | 有 top-down feedback、无 paired flux |
| `cifar10_cocurrent_selfboundary.yaml` | 相同 boundary/content/operator 的同向关键对照 |
| `cifar10_countercurrent_selfboundary.yaml` | 主 Countercurrent 模型 |
| `cifar10_countercurrent_null.yaml` | `C_L=0` topology ablation |
| `cifar10_countercurrent_learnedz.yaml` | sample-independent learned boundary ablation |

标准设置为 `channels=64, L=4, T=3`。主 Countercurrent 与 Co-current 都是 673,418 参数、
186,399,232 Conv/Linear MACs；两者完全相等。Forward recurrent 为 668,362 参数、
186,386,944 MACs，控制“多算几轮”本身；8-block 单流模型为 668,362 参数，作为近似参数匹配对照。

## 测试

```bash
conda run -n 4dflow python -m unittest discover -s tests -v
```

测试覆盖 soft boundary、无 GT leakage、每轮 source clamp、`t=0` 纯 forward、conductance
范围、paired-exchange 代数、双向局部影响、reverse-off、所有 state shape，以及 CC/Co 的参数量和
MAC 严格一致性。

## 训练与评估

先跑完整链路的合成数据 smoke test：

```bash
conda run -n 4dflow python -m countercurrent_nn.train \
  --config countercurrent_nn/configs/cifar10_countercurrent_selfboundary.yaml \
  --debug --epochs 1 --synthetic-data \
  --limit-train-batches 1 --limit-eval-batches 1 \
  --output-dir /tmp/countercurrent_smoke
```

CIFAR-10 主实验：

```bash
conda run -n 4dflow python -m countercurrent_nn.train \
  --config countercurrent_nn/configs/cifar10_countercurrent_selfboundary.yaml \
  --seed 0
```

Countercurrent 与 Co-current 应按相同 seeds 成对运行。探索使用 `0,1,2`，正式结果使用
`0,1,2,3,4` 并报告 mean ± std。

训练结果除 accuracy、MACs、latency 和 peak memory 外，还会记录：

- `t=0..T` accuracy、confidence、predictive entropy；
- Error Correction Rate 与 Error Amplification Rate；
- exchange magnitude、discrepancy、local exchange energy、H/C norms；
- 同一个已训练模型在 `Q=0` 时的 reverse-off accuracy drop。

单独执行 reverse-off：

```bash
conda run -n 4dflow python -m countercurrent_nn.analysis.reverse_off_eval \
  countercurrent_nn/results/RUN/best.pt
```

绘图与确认偏差分析：

```bash
conda run -n 4dflow python -m countercurrent_nn.analysis.plot_iteration_accuracy RUN/diagnostics.pt
conda run -n 4dflow python -m countercurrent_nn.analysis.plot_entropy RUN/diagnostics.pt
conda run -n 4dflow python -m countercurrent_nn.analysis.plot_exchange RUN/diagnostics.pt
conda run -n 4dflow python -m countercurrent_nn.analysis.plot_discrepancy RUN/diagnostics.pt
conda run -n 4dflow python -m countercurrent_nn.analysis.error_correction_analysis RUN/diagnostics.pt
```

当前仓库只交付架构、实验协议与可运行验证，没有把 synthetic smoke 指标包装成 CIFAR 科研结果。
