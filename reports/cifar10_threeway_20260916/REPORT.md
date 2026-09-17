# CIFAR-10 三组实验分析与下一轮设计

分析日期：2026-09-16。依据三份权威规范、当前 v2 实现、三组完整训练日志与 best checkpoint。此次工作只增加分析脚本和报告，没有修改模型、优化器或原实验文件，没有启动训练。

## 1. 当前能支持什么结论

**迭代交换存在实际纠错收益，但尚未验证 opposing-flow 优于 same-direction flow，也没有取得相对参数匹配 CNN 的精度或计算效率优势。** 这是值得继续定位机制的第一轮结果，不能直接作为核心假说已成立的证据。

三组使用相同 CIFAR-10 划分（45,000 训练 / 5,000 验证 / 10,000 测试）、seed=0、200 epochs、SGD、全局 batch=128、四卡每卡 batch=32；学习率、增强及其他训练配置一致。普通 CNN 实际是 **8 个 transport block 的参数匹配版本**；两个 coupled 模型均为 L=4、T=3，参数配置仅 topology 名称不同。

| 模型 | 最佳验证准确率 | 对应测试准确率 | 测试 CE | 参数量 | 每张图片 MACs | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CNN，参数匹配 | 89.02% | 88.04% | 0.4323 | 668,362 | 51.315 M | 193 |
| Co-current v2 | 88.90% | 88.31% | 0.4213 | 673,418 | 243.022 M | 195 |
| Countercurrent v2 | 88.34% | 87.96% | 0.4409 | 673,418 | 243.022 M | 194 |

Countercurrent 比 Co-current 低 **0.35 个百分点（35 张测试图片）**，比 CNN 低 **0.08 个百分点（8 张）**。两种 coupled 模型比 CNN 多约 0.76% 参数，但 Conv2d+Linear MACs 为 **4.74 倍**。参数匹配成立，计算量匹配不成立；单 seed 的微小差异不能解释为稳定优劣，也不能据此否定整个思想。仅凭三个总体准确率无法计算配对显著性，更不能估计训练 seed 方差。

三组 200 epochs 的记录完整，训练及验证 loss 均有限，已保存诊断 tensor 没有 NaN/Inf。末轮训练准确率分别为 98.70%、97.63%、97.86%；后期训练 loss 继续下降，而验证提升有限。Countercurrent 不是明显没有收敛的情形；目前没有充分依据只延长训练或加大 T。

![训练曲线和机制诊断](comparison.png)

## 2. 交换是否参与预测、迭代是否真的纠错

以下来自原实验 **完整 10,000 张测试集**的 `summary.json`。

| 模型 | t=0 / reverse-off | t=1 | t=2 | t=3 | 相比 t=0 净提升 |
|---|---:|---:|---:|---:|---:|
| Co-current | 86.12% | 88.08% | 88.34% | 88.31% | +2.19 pp |
| Countercurrent | 85.29% | 87.45% | 87.92% | 87.96% | +2.67 pp |

`reverse-off` 设置 Q=0，是对已经联合训练的模型做推理消融；它不是单独训练的公平前馈 baseline。因此 2.67 pp 能说明当前预测依赖交换计算，不能证明“逆流架构比普通 CNN 高 2.67 pp”，也不能单独证明收益来自有意义的 target hypothesis。

| 纠错指标 | Co-current | Countercurrent |
|---|---:|---:|
| 初始错判数 | 1,388 | 1,471 |
| 错→对 | 396 | 421 |
| 对→错 | 177 | 154 |
| 净增加正确数 | 219 | 267 |
| ECR：初始错判被纠正比例 | 28.53% | 28.62% |
| 原本正确样本被改错比例 | 2.06% | 1.81% |
| EAR：原始错误类别置信度增加 >0.1 | 26.30% | 23.59% |

Countercurrent 净纠错更多、误伤更少，是可以跟进的探索性信号；但它初始错误也更多，归一化 ECR 几乎相同，不能把 421 vs 396 当成强机制优势。当前 EAR 的代码定义不要求最终仍然错判，不能把 EAR 和 ECR 当作互补比例。

Countercurrent 从 t=0 到 t=3 补回了相对 Co-current 初始差距中的 0.48 pp，但仍然落后 0.35 pp。第三轮只增加 0.04 pp，而平均置信度仍从 94.64% 升至 95.11%；这提示应检查假设反馈的校准，而不是直接加迭代。平均置信度减准确率不是 ECE，后面的验证集诊断另行计算分箱 ECE。

## 3. 一个已定位的优化器问题：weight decay 的吸引点

当前 `models/exchange.py` 定义 `gamma = 0.49 * sigmoid(logit_gamma)`，初始化 `logit_gamma=-2.2`，对应 gamma≈0.0489。`engine.py:build_optimizer` 将所有参数直接放入同一 weight-decay 参数组，因而也衰减 `logit_gamma`。

对于负的 logit，衰减到 0 会使 gamma **增大到 0.245**，并不是让交换减弱。按当前实际 SGD、momentum、warmup/cosine 和已训练步数，令任务梯度恒为零，只保留 weight decay 做标量反事实计算，最终 gamma 仍会达到约 **0.245000**。这里运行的是标量更新公式，不是训练一个网络。

实际 best checkpoint 的各层 gamma 均值：

| cell | Co-current | Countercurrent |
|---|---:|---:|
| 0 | 0.24422 | 0.24448 |
| 1 | 0.24348 | 0.24318 |
| 2 | 0.24387 | 0.24331 |
| 3 | 0.23533 | 0.23537 |

因此，不能将“gamma 从约 0.05 学到约 0.24”直接当作自适应交换被验证的证据。它与明确的正则化偏置相符；但这也不证明去掉该偏置一定能提高精度，需要重新训练做对照，不能直接把已训练模型的 gamma 重置到初始值来判定。

第一项建议：仅将 `exchange.*.logit_gamma` 分到 `weight_decay=0` 的参数组，其余参数、初始化与训练设置保持一致。另一种可选设计是 `gamma=0.49*sigmoid(-2.2+delta)`，delta 初始化为 0 并对 delta 衰减，这样正则化吸引点对应初始交换强度。两种方案择一做主实验即可。

## 4. 验证集冻结权重干预

已在固定的 **完整 5,000 张验证集**上重新运行三组 best checkpoint，统一使用 CPU、eval 模式、batch=32、无梯度，不修改权重。1024 张子集仅用于确定是否扩大诊断，下面所有结论使用完整验证集。CPU 复测 Co-current=88.92%、Countercurrent=88.32%，与保存的 GPU 验证结果分别相差 +1 / -1 张图片；本表各项干预都与同一 CPU 复测基线比较，不混用二者。这种微小复测差异也提醒我们不要过度解读一两张图片的变化。

| 推理中的 boundary | Co-current 准确率 | Countercurrent 准确率 | Co-current CE | Countercurrent CE |
|---|---:|---:|---:|---:|
| 原始 self-generated hypothesis | 88.92% | 88.32% | 0.4236 | 0.4178 |
| 每类概率均为 1/10，仍使用已训练 prototypes/projector | 88.52% | 88.32% | 0.3568 | 0.3472 |
| boundary 向量直接设为 0，保留 G 与交换 | 88.42% | 87.98% | 0.3617 | 0.3518 |
| batch 内循环错配另一张图的 hypothesis | 87.12% | 86.70% | 0.3959 | 0.3911 |
| 仅 boundary 使用 `softmax(logits/2)` | 88.66% | 88.34% | 0.3781 | 0.3722 |

**最值得注意的是：Countercurrent 的真实 hypothesis 相比均匀 hypothesis，没有取得总体准确率收益。** 均匀化后有 30 张由错变对、30 张由对变错，因而准确率持平，不意味着两种输出相同。错配 hypothesis 又使准确率降低 1.62 pp，说明模型会响应假设内容，但“对内容敏感”和“动态正确对齐的内容在总体上有正贡献”是不同命题。

均匀 / 零 boundary 仍保留 reverse transport 和 exchange，不能等同于 reverse-off。常量 boundary 的 t=1、2、3 logits 完全一致，说明在当前反复从 x 重建内部状态的调度下，动态 boundary 被拿掉后，后续轮次没有额外计算状态可继续更新。

校准方面，Countercurrent 的 15-bin ECE 从原始 **7.00%** 降至均匀 boundary 的 **2.08%**，tau=2 时为 **4.61%**；Co-current 对应为 6.61%、1.83%、4.37%。原始 self 模式在两个模型中都呈现从 t=1 到 t=3 CE 变差、ECE 变大的情况，而准确率变化很小。这里的 ECE 是实际按置信度分箱计算，结果支持存在过度确信问题。

tau=2 给 Countercurrent 只带来 **+0.02 pp，即净多 1 张正确图片**（13 张改善、12 张变坏），且 Co-current 下降 0.26 pp。因此它目前是改善 CE / 校准的诊断线索，**不是已经证实能提高分类准确率的方案**；1024 张初筛中更显眼的提升没有在完整验证集上保持。

另行标记的 GT oracle 分析得到 Co-current=95.24%、Countercurrent=95.10%。它直接向边界注入验证标签，属于有标签信息的诊断，不能作为正常分类结果、可实现上界或“改善预测就一定能达到 95%”的证据。

本次所有模式的 5,000 张 logits 均为有限值，恒定 boundary 的逐轮一致性检查通过。冻结权重干预存在训练/推理分布改变，因此仍需从头训练 null / learned constant 对照来分离架构收益。

## 5. 层间交换和当前更新方式的边界

现有 `diagnostics.pt` 只保存 **16 张测试图片**的详细状态，不能把其中的小样本准确率、ECR 等替代完整测试指标。

在这 16 张图片的最终迭代中，最后一个 cell 占各层 Q 平方范数总和的 84.20%（Countercurrent）和 83.84%（Co-current）。但归一化后 Countercurrent 的 `||Q||/||H_bar||` 为约 `[0.192, 0.178, 0.161, 0.182]`，Co-current 为 `[0.187, 0.173, 0.158, 0.179]`。最后层的绝对 Q 大，部分来自状态尺度大；不能据此声称前几层没有交换。当前没有看到两种拓扑在这些诊断中出现明显分化，需要对更多样本、多个 seed 检验 distributed discrepancy。

v2 对应规范 §15 的有限 two-sweep 近似：先独立计算 F/G proposals，进行成对交换，再将修正后的 C 用于 forward resweep。第一阶段的 `C+Q` 确实影响输出；但它没有继续穿过相邻 G block 向逆流上游传播，第二阶段的 C 输出也不会作为下一轮的内部 C 状态保留。

在单个 cell 内，这个调度的最终 forward 更新可以展开为：

`H_out = (1-gamma)*H_bar + gamma*(1-gamma)*C_proposal + gamma^2*H_proposal`。

因此，即使输入的是常量 target boundary，交换仍可通过 forward 缩放、额外 proposal 路径和 G 变换改变输出。这也是需要 null / constant / shuffled boundary 消融，而不能只做 reverse-off 的原因。v2 结果只能检验这个有限近似，不能直接代表充分传播的 reciprocal lattice 已被检验。

## 6. 下一轮建议和判断标准

第一轮改进应优先使用不增加通道、深度或迭代次数的方案，保持 source 每轮重新注入、soft self-generated boundary、正 conductance 和同一个 Q 的 paired exchange。

1. **纠正 conductance 的正则化偏置。** 采用上面的 gamma 参数单独不衰减方案，两种拓扑同时应用。
2. **给初始 hypothesis 加轻量监督。** 候选损失 `CE(logits_T,y) + 0.2*CE(logits_0,y)`，0.2 是建议起点而非已验证最优值。当前仅监督最终输出，初始假设质量未被直接约束；这项改动检验更好的初始判断能否让后续交换用于进一步纠错。它属于规范 §23 的可选 intermediate supervision。标签只进入 loss；主模型推理仍不接受标签。初始 logits 已经计算，推理参数量和 MACs 不增加，但训练目标变了，不能承诺一定改善最终输出。

将这两项做以下 2×2 消融，候选的轻量 v3 是二者结合：

| 设置 | gamma 参数 weight decay | 初始 CE 权重 | boundary 温度 |
|---|---:|---:|---:|
| 原 v2 | 0.0005 | 0 | 1 |
| 仅修正 gamma 衰减 | 0 | 0 | 1 |
| 仅加入初始监督 | 0.0005 | 0.2 | 1 |
| 两者结合，候选 v3 | 0 | 0.2 | 1 |

**每个设置都运行 Co-current 和 Countercurrent**，保持相同训练预算；原 v2 seed=0 可复用，所以首轮筛查新增 6 个训练。先按固定验证集筛查，再对选中的方案及原始对照补 seed=1、2，条件允许扩展到 5 个 seed。若初始监督只提高 t=0 而损害最终精度，也应如实判为没有达到目标，不能只展示初始准确率。

完整验证集上的 tau=2 并未显示明确准确率改善，因此不放进第一轮主方案。可在下一阶段单独检验它是否有助校准或错误放大；若使用，只在 boundary 构造里做 `p=softmax(logits/tau)`，保留最终分类 logits。不要同时增加第三个因素，以免无法定位收益来源。所有调参使用验证集；本报告里的测试集迭代曲线仅解释已有结果，不用来选择 T 或其他超参数。

后续必须补全规范中的 Forward-only Recurrent、Feedback-Fusion、从头训练的 Null-boundary / Learned-Z 对照。除参数量外还要报告计算量和实际速度，检验收益是否仅来自额外计算。冻结模型的边界干预不能替代这些训练对照。

若这些控制后 Countercurrent 仍不优于 Co-current，下一步才考虑更强的结构修订：让修正后的 reverse state 继续沿 G 传播，或实现规范 §16 的有限步同步 lattice。这里需重新明确更新调度，并对 Co-current 做完全相同的调度升级，验证两种流向以外的变量一致；不能将这一修改与简单超参数修正混为一谈。

按照规范 §38，目前已有 reverse-off 和真实纠错的信号，但尚未达到多个 seed 下 CC>Co 的性能标准，也未证明逆流特有的 distributed exchange 优势。下一轮更有说服力的目标是：CC 相对 Co 的差异在多个 seed 为正、真实 hypothesis 比常量/错配边界更有效、错误放大减少，并在额外计算对照下仍保留优势。达到这些条件后再扩展 CIFAR-100，会比直接换更大数据集更容易解释结果。

## 7. 复现与产物

原实验目录均在 `countercurrent_nn/results/`：

- `cifar10_single_parammatched_4gpu_seed0`
- `cifar10_cocurrent_v2_4gpu_seed0`
- `cifar10_countercurrent_v2_4gpu_seed0`

本次分析脚本为 `countercurrent_nn/analysis/compare_three_runs.py`。完整验证集推理命令（CPU，无训练）：

```bash
cd /localhome/zhanghs/Countercurrent
MPLCONFIGDIR=/tmp/countercurrent_mpl_cache \
  /home/zhanghs/miniconda3/envs/4dflow/bin/python \
  -m countercurrent_nn.analysis.compare_three_runs \
  --probe-samples 5000 \
  --output reports/cifar10_threeway_20260916/full_validation
```

`analysis.json` 保存原实验汇总、派生纠错统计和 1024 张初筛；`full_validation/analysis.json` 保存完整验证集干预结果；对应 `validation_probe_predictions.pt` 保存各模式逐迭代 logits 和 labels，支持进一步配对分析。`comparison.png` / `comparison.pdf` 为训练及机制对比图。Oracle 使用标签仅用于明确标记的分析路径，不能报告为正常模型性能或可实现的精度上界。
