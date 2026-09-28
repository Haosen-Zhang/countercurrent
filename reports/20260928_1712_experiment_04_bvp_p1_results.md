# 第 4 次实验：BVP-P1 Co-current 与 Countercurrent

- 时间：2026-09-28 17:12（Asia/Shanghai）
- 数据集：CIFAR-10
- 实验编号：04
- 代码版本：`898ad3a`（修复 adaptive BVP refinement 的跨 batch 统计）
- 状态：seed 0 成对实验完成

编号按主要成对实验阶段计算：V2 三组基线为第 1 次，V3 为第 2 次，V4 为第 3 次，
本次 BVP-P1 为第 4 次。

## 1. 实验目的

本次实验检验真正的 two-point boundary-value solver 是否能让 Countercurrent 的 opposing flow
优于使用相同 boundary、exchange law、优化协议和模型规模的 Co-current。BVP-P1 保留跨 sweep
的 forward state，以残差判据提前停止；target boundary 同时依赖 soft class hypothesis 与
stop-gradient 的实例特征。

## 2. 对比协议

| 项目 | Co-current P1 | Countercurrent P1 |
|---|---|---|
| 配置 | `cifar10_bvp_cocurrent_p1.yaml` | `cifar10_bvp_countercurrent_p1.yaml` |
| 模型 | `bvp_cocurrent` | `bvp_countercurrent` |
| C 流向 | 0 → L | L → 0 |
| 参数量 | 681,740 | 681,740 |
| Conv/Linear MACs | 358,065,792 | 358,065,792 |
| 最大求解步数 | 8 | 8 |
| 停止容差 | 1e-3 | 1e-3 |

共同训练设置：CIFAR-10 固定 45k/5k/10k split，SGD，初始学习率 0.1，momentum 0.9，
weight decay 5e-4，5 epochs warmup + cosine，200 epochs，全局 batch 128，两张 GPU，seed 0。
本次只有一个 seed，且 `deterministic=false`。

训练结束后的第一次 refinement evaluation 因不同 batch 的提前停止步数不同而发生张量长度
不匹配。修复后，已经收敛的 batch 用最终 logits 填充到 `solve_steps + 1` 个统计点；模型求解、
最终预测和 checkpoint 均未改变。随后重新加载各自 best checkpoint 完成测试。

## 3. 主要结果

| 指标 | Co-current P1 | Countercurrent P1 | CC − Co |
|---|---:|---:|---:|
| 最佳 validation accuracy | **88.24%** | 88.10% | -0.14 pp |
| Test accuracy | **87.65%** | 87.38% | -0.27 pp |
| Test CE | 0.4164 | **0.4105** | -0.0059 |
| t=0 / reverse-off accuracy | **87.30%** | 86.79% | -0.51 pp |
| Refinement gain | +0.35 pp | **+0.59 pp** | +0.24 pp |
| Error correction rate | 11.50% | **14.08%** | +2.58 pp |
| Error amplification rate | **11.65%** | 12.72% | +1.07 pp |

逐步测试准确率：

```text
Co-current:     87.30, 87.28, 87.32, 87.55, 87.60, 87.61, 87.63, 87.65, 87.65
Countercurrent: 86.79, 86.78, 86.93, 87.29, 87.31, 87.30, 87.33, 87.40, 87.38
```

Countercurrent 的 refinement 增益更大，但其 t=0 forward prediction 低 0.51 pp，因此最终仍低于
Co-current 0.27 pp。Countercurrent 的最低 CE 表明它的概率输出并非全面更差，但本次没有计算
ECE，不能据此宣称校准更好。

## 4. 完整测试集配对检验

在相同的 10,000 张测试图片上逐样本比较两个 best checkpoint：

| 配对结果 | 样本数 |
|---|---:|
| 两者都正确 | 8,249 |
| 仅 Co-current 正确 | 516 |
| 仅 Countercurrent 正确 | 489 |
| 两者都错误 | 746 |

- Co-current − Countercurrent：+0.27 pp；
- paired bootstrap 95% CI：`[-0.35, +0.89]` pp；
- exact McNemar：`p=0.412`。

置信区间跨过 0，且 McNemar 检验不显著。固定 checkpoint 的测试样本证据不能证明任一拓扑
更优；该区间还没有包含训练 seed 方差。

## 5. 求解器收敛诊断

以 batch size 64 遍历完整测试集，共 157 个 batch：

| 指标 | Co-current P1 | Countercurrent P1 |
|---|---:|---:|
| 8 步内达到 residual < 1e-3 | 40.8% | 31.8% |
| 平均实际求解步数 | 6.78 | 7.04 |
| 第 5 步停止 | 64 batch | 50 batch |
| 达到最大 8 步 | 93 batch | 107 batch |
| 最终 residual mean | 0.00170 | 0.00177 |
| 最终 residual median | 0.00189 | 0.00196 |
| 最终 residual max | 0.00340 | 0.00304 |

两种拓扑的大多数 batch 都没有在 8 步内达到规定容差，Countercurrent 的收敛率更低。因此当前
结果更准确的描述是“最多 8 步的截断 BVP refinement”，尚不能视为充分收敛的 two-point BVP
解。这是当前实验最重要的结构性限制。

## 6. 当前结论

### 可以支持

1. BVP exchange 对两种拓扑都产生了非零的测试收益。
2. Countercurrent 对 exchange 的依赖更强：reverse-off drop 为 0.59 pp，Co-current 为 0.35 pp。
3. Countercurrent 的错误纠正率和 refinement 净增益更高，说明 opposing flow 确实参与了推理。
4. Co/Counter 参数量和所统计 MACs 完全一致，本次同 seed 对比协议公平。

### 不能支持

1. **不能证明 Countercurrent 优于 Co-current。** 最终准确率低 0.27 pp，配对差异不显著。
2. **不能宣称 BVP 已经稳定求解到固定点。** CC 仅 31.8% 的 batch 在 8 步内达到容差。
3. **不能据一个 seed 得出架构排序。** 还缺少训练 seed 方差。
4. **不能据此扩展论文结论到 CIFAR-100。** 当前应先解决收敛和重复性。

本次结果比 V4 更清楚地证明 exchange 具有实际作用，但还没有验证核心主张
“opposing transport 比 same-direction transport 更有效”。

## 7. 下一步

1. 使用现有 checkpoint 做 `solve_steps=8/12/16/24` 的纯推理扫描，记录 accuracy、CE、实际步数、
   residual 和计算开销，确认充分收敛是否改变 CC/Co 排序。
2. 检查当前 residual 非单调的原因，并比较 damping、Anderson acceleration 或更合适的停止判据；
   在求解器稳定前不应直接扩大训练规模。
3. 固定选定的 solver 设置后补 seed 1、2；若 `CC-Co` 方向稳定，再扩到 5 seeds。
4. 继续留在 CIFAR-10。只有收敛、多个 seed 和关键机制指标稳定后，再迁移到 CIFAR-100。

## 8. 结果位置

- Co-current：`countercurrent_nn/results/cifar10_bvp_cocurrent_p1_2gpu_seed0/summary.json`
- Countercurrent：`countercurrent_nn/results/cifar10_bvp_countercurrent_p1_2gpu_seed0/summary.json`
- 两组目录均包含 `best.pt`、`last.pt`、`metrics.jsonl`、`diagnostics.pt/json` 和 `summary.json`。
