# Experiment 05：V5-C 20-epoch CC/Co Pilot（2026-09-29—30）

本实验在正式 200 epochs 训练前，比较严格匹配的 V5-C Countercurrent 与 Co-current，
并检查 solver、canonical exchange 和 target-boundary semantics。实验基于提交 `a0a6132`；
没有启动 200 epochs，也没有进入 CIFAR-100。

## 1. 设置与公平性

| 项目 | Countercurrent | Co-current |
|---|---:|---:|
| Dataset | CIFAR-10 | CIFAR-10 |
| Seed | 0 | 0 |
| Epochs | 20 | 20 |
| Global batch | 128 | 128 |
| GPUs | 2 | 2 |
| Transport depth | 4 | 4 |
| Train inner steps | 8 | 8 |
| Eval max steps | 24 | 24 |
| Parameters | 812,270 | 812,270 |
| Max Conv/Linear MACs | 1,010,318,976 | 1,010,318,976 |

两组使用相同 F/G、canonical exchange、boundary、head 和训练协议；唯一结构差异是 C 的传播
方向。训练固定执行 8 个 inner steps，因此训练计算预算一致。

## 2. 分类结果

| 指标 | Countercurrent | Co-current | Co − CC |
|---|---:|---:|---:|
| Best validation accuracy | 78.80% | 78.82% | +0.02 pp |
| Test accuracy | 77.72% | 77.73% | +0.01 pp |
| Reverse-off test accuracy | 75.62% | 74.73% | — |
| Reverse accuracy gain | +2.10 pp | +3.00 pp | +0.90 pp |
| Best epoch | 20 | 19 | — |

最终准确率在实验分辨率内完全相同。Reverse stream 对两组预测均有贡献，但 Co-current 的
gain 更大，因此该 gain 不能作为 opposing transport geometry 的证据。

## 3. Solver 与 canonical direction

| 机制指标 | Countercurrent | Co-current | Gate |
|---|---:|---:|---|
| Diagnostic convergence | 16/16 | 16/16 | Pass |
| Final mean equation residual | 0.000784 | 0.000841 | Pass |
| Typical eval solve steps | 14–15 | 15–16 | Pass；Co 稍慢 |
| Final mean `abs(cos(D,u_H))` | 0.328 | 0.292 | Pass |
| Final max `abs(cos(D,u_H))` | 0.553 | 0.457 | Pass |

两组都稳定达到 `r_eq < 1e-3`，并且 discrepancy 没有退化为与 H 同向的简单 scaling。

## 4. Boundary collapse

| 指标 | Countercurrent | Co-current |
|---|---:|---:|
| Epoch-1 `R_class` | 0.2866 | 0.2923 |
| Final `R_class` | 0.00631 | 0.00720 |
| Final diagnostic `R_base` | 0.3026 | 0.3447 |
| Final diagnostic `R_class` | 0.00653 | 0.00729 |
| Final diagnostic `R_instance` | 0.9519 | 0.9376 |
| Learned `instance_scale` | 1.527 | 1.459 |
| Learned `class_scale` | 1.038 | 1.037 |

`R_class` 在两组中都下降约 40–45 倍。Class scalar 本身没有缩小；instance scalar 从 0.1
增长到约 1.5，同时 class embedding RMS 降到约 0.05。模型通过 instance branch 建立了
更容易的 image-conditioned feedback shortcut。

## 5. Frozen-checkpoint boundary interventions

以下干预在相同的前 2,560 张 CIFAR-10 test images 上执行：

| Boundary variant | Countercurrent | Co-current |
|---|---:|---:|
| Real hypothesis | 77.50% | 77.27% |
| Uniform hypothesis | 77.50% | 77.27% |
| Shuffled hypothesis | 77.54% | 77.27% |
| Class-off | 77.50% | 77.27% |
| Instance-off | 76.17% | 75.47% |
| Base-only | 76.17% | 75.55% |
| Zero boundary | 75.90% | 75.39% |

Real、uniform、shuffled 与 class-off 在数值上没有有意义的差异；移除 instance 则下降
约 1.3–1.8 pp。这直接确认当前 reverse gain 来自 instance feedback，而不是 class
hypothesis。

## 6. Go / No-Go

| Gate | 结果 |
|---|---|
| Solver 在 24 步内收敛 | Pass |
| Target hypothesis 有实际语义贡献 | **Fail** |
| Countercurrent 稳定优于 Co-current | **Fail** |
| Exchange 不退化为 scaling | Pass |

结论是 **No-Go**：不得据此启动 V5-C 200 epochs，也不应直接切换 CIFAR-100。当前实验
验证了 persistent solver 和 canonical exchange 的数值可行性，但没有验证 class-conditioned
countercurrent inference 或 opposing topology 优势。

## 7. 可复现材料

- 本目录中的 `countercurrent_*` 与 `cocurrent_*` 文件保存配置、逐 epoch metrics、最终
  diagnostics、profile 和 summary；checkpoint/tensor 文件因体积与仓库规则不上传。
- `boundary_interventions_2560.json` 保存两组冻结 checkpoint 干预的原始 loss/accuracy。
- 可使用 `python -m countercurrent_nn.analysis.v5_boundary_intervention CHECKPOINT ...`
  重新运行干预。
- 下一阶段实现与决策流程见仓库根目录的
  `COUNTERCURRENT_V5C_NEXT_VALIDATION_PLAN.md`。

两次 wall time 不用于比较模型速度，因为运行期间机器负载不同。架构 MAC 完全一致；实际
推理成本应通过同机、连续、相同 batch 的专门 benchmark 比较。
