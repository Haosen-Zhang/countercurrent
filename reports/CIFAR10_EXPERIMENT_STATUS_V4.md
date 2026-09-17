# Countercurrent CIFAR-10 实验阶段报告：V2–V4

更新时间：2026-09-17

本报告记录当前代码、已完成实验、结果边界以及下一阶段假设。设计定义以仓库根目录的
`COUNTERCURRENT_REIMPLEMENT_PROMPT.md`、`COUNTERCURRENT_CORE_IDEA.md` 和
`COUNTERCURRENT_CNN_POC_EXPERIMENT.md` 为准。

## 1. 当前研究问题

核心问题是：source evidence 与模型自己生成的 target hypothesis 从深度轴两端进入，沿相反方向
传播，并通过逐层 paired exchange 互相修正，是否能比相同参数、相同 boundary 和相同 exchange law
的 Co-current 更有效。

局部交换保持：

\[
D_l=\bar H_{l+1}-\bar C_l,
\qquad
Q_l=\gamma_l\odot D_l,
\]

\[
H_{l+1}=\bar H_{l+1}-Q_l,
\qquad
C_l=\bar C_l+Q_l,
\qquad
0<\gamma_l<0.5.
\]

主模型 inference 不接收 GT；标签只进入训练 loss。每轮 refinement 都重新计算 `stem(x)`，保持
source evidence hard clamp。

## 2. 当前模型和实验协议

- 数据集：CIFAR-10，固定 45k train / 5k validation / 10k test。
- lattice：`L=4, T=3, channels=64`，所有 H/C 状态为 `[B,64,8,8]`。
- F/G：相互独立的 residual CNN block，`x + 0.1*f(x)`，GroupNorm + SiLU。
- boundary：`softmax(logits) @ class_prototypes`，再经 `Linear(64,64)` 并广播到 8×8。
- head：global average pooling + `Linear(64,10)`。
- 参数：673,418；Conv/Linear MACs：243,022,336；近似 FLOPs：486,044,672。
- 优化：SGD，lr=0.1，momentum=0.9，200 epochs，5 epochs warmup + cosine。
- 全局 batch size：128。

参数匹配普通 CNN 有 668,362 参数和 51,315,328 MACs。因此 coupled 模型参数只多约 0.76%，
但所统计 MACs 是 4.74 倍。

## 3. 版本演进

### V2：proposal → parallel reconciliation → corrected forward resweep

V2 先独立生成完整 H proposal 和 C proposal，再在各层执行 paired exchange。第一遍修正后的 C
会被 forward resweep 使用，但某层 `C+Q` 不会继续通过相邻 G block。

### V3：训练消融

V3 保留 V2 inference，取消 conductance logit 的 weight decay，并使用：

\[
L=CE(s^T,y)+0.2CE(s^0,y).
\]

结果显示模型把主要任务交给初始 forward prediction，exchange 几乎关闭。因此初始辅助 CE
不再用于 V4 主实验。

### V4：sequential reverse reconciliation

V4 在 reverse sweep 中逐 cell 执行 transport 和 exchange：

```text
C[L] = target boundary
for l = L-1 ... 0:
    C_bar[l] = G[l](C[l+1])
    Q[l] = gamma[l] * (H_prop[l+1] - C_bar[l])
    H_rec[l+1] = H_prop[l+1] - Q[l]
    C[l] = C_bar[l] + Q[l]
             └── immediately enters G[l-1]
```

Co-current 使用相同调度，传播方向为 `0 -> L`。之后两者都执行 corrected forward resweep。
V4 使用最终 CE，不使用初始辅助 CE；conductance logit 的 weight decay 为 0，其他参数仍为
5e-4。

代码通过 checkpoint 内 `_inference_version` 区分调度：V2/V3 为 2，V4 为 3。旧 checkpoint
会执行原调度，训练入口拒绝跨 inference version 续训。

## 4. 已完成的 seed-0 结果

| 模型 | 最佳验证准确率 | 测试准确率 | 测试 CE | t=0 | t=3 / final | reverse-off drop |
|---|---:|---:|---:|---:|---:|---:|
| 参数匹配 CNN | 89.02% | 88.04% | 0.4323 | — | 88.04% | — |
| Co-current V2 | 88.90% | **88.31%** | 0.4213 | 86.12% | 88.31% | 2.19 pp |
| Countercurrent V2 | 88.34% | 87.96% | 0.4409 | 85.29% | 87.96% | 2.67 pp |
| Co-current V3 | 88.60% | 87.20% | 0.4250 | 87.29% | 87.20% | -0.09 pp |
| Countercurrent V3 | 88.18% | 87.81% | **0.4100** | 87.82% | 87.81% | -0.01 pp |
| Co-current V4 | 88.04% | 87.17% | 0.4444 | 84.63% | 87.17% | 2.54 pp |
| Countercurrent V4 | **88.30%** | **87.62%** | 0.4244 | **85.49%** | **87.62%** | 2.13 pp |

V2 使用 4 GPU、V3 使用 1 GPU、V4 使用 2 GPU；全局 batch 都是 128。所有结果只有 seed=0，
且配置为 `deterministic: false`。同版本内 CC/Co 协议匹配，跨版本的细小差异不能当作严格配对
架构效应。

## 5. V4 结果分析

V4 的 Countercurrent 在 validation 和 test 上分别比 Co-current 高 0.26 和 0.45 pp。这是当前
第一次出现 CC 在两个 split 上都领先 Co，方向积极，但证据仍弱。

对 10,000 张相同测试图片做固定 checkpoint 的逐样本配对比较：

- 仅 Countercurrent 正确：522；
- 仅 Co-current 正确：476；
- 准确率差：+0.46 pp（CPU 复测）；
- 配对 95% CI：[-0.16, +1.08] pp；
- exact McNemar p=0.154。

这个区间只表示固定 checkpoint 下的测试样本不确定性，不包含训练 seed 方差。因此不能宣称
Countercurrent 已显著优于 Co-current。

### 5.1 真实纠错恢复，但 CC 的迭代收益没有超过 Co

```text
Co-current V4:     84.63 -> 87.07 -> 87.13 -> 87.17
Countercurrent V4: 85.49 -> 87.64 -> 87.63 -> 87.62
```

| 指标 | Co-current V4 | Countercurrent V4 |
|---|---:|---:|
| 初始错判 | 1,537 | 1,451 |
| 错→对 | 432 | 344 |
| 对→错 | 178 | 131 |
| 净增加正确数 | 254 | 213 |
| ECR | 28.11% | 23.71% |
| EAR | 14.12% | 14.20% |

V4 Countercurrent 的最终领先主要来自更强的 t=0 forward state；它的 refinement 净收益和 ECR
反而低于 Co-current。V4 支持“exchange 参与有效预测”，尚不支持“opposing flow 的纠错比
same-direction flow 更强”。

Countercurrent 的第一轮达到最高准确率 87.64%，后两轮轻微下降。16 张详细诊断样本的 H-state
相对变化约为 `18.0% -> 0.31% -> 0.015%`，说明当前动力学在一轮后已基本固定。T=3 对
Countercurrent 没有观测到额外收益。

### 5.2 Conductance 和 distributed flux

V4 Countercurrent best checkpoint 的逐层 mean gamma：

```text
[0.0518, 0.0458, 0.0429, 0.0319]
```

Co-current 为：

```text
[0.0503, 0.0451, 0.0422, 0.0317]
```

V4 没有依赖 V2 中 weight decay 将负 logit 拉向 0、把 gamma 人工推向 0.245 的效应。虽然 gamma
较小，reverse-off 仍下降约 2%，证明交换不是 V3 那样的空路径。

在 16 张诊断样本上，V4 Countercurrent 最终 Q 平方范数的逐层占比约为：

```text
[19.0%, 15.3%, 14.3%, 51.4%]
```

V2 最后一层约占 84.2%。这与 sequential reverse propagation 使 flux 更分布化的预期相符，
但样本数太少，只能作为机制线索。

### 5.3 Nonzero boundary 重要，class-hypothesis 语义贡献仍弱

冻结 V4 best checkpoint，在完整 5,000 张 validation split 上干预 boundary：

| Boundary | Co-current V4 | Countercurrent V4 |
|---|---:|---:|
| 原始 self-generated hypothesis | 88.02% | **88.30%** |
| uniform class probability | 88.06% | 88.22% |
| batch 内错配另一张图片的 hypothesis | 88.08% | 88.18% |
| zero boundary | 84.66% | 86.38% |
| boundary temperature=2 | 88.10% | 88.28% |

Countercurrent 的真实 hypothesis 只比 uniform 高 0.08 pp、比 shuffled 高 0.12 pp。Zero boundary
明显下降，说明 reverse reservoir 很重要；但 sample-specific class semantics 的额外作用仍然很小。
冻结干预改变了训练/推理分布，不能替代从头训练的 null/learned-Z control。

## 6. 当前结论与假设状态

### 已得到支持

1. Layerwise paired exchange 可以参与推理并产生真实错→对样本；reverse-off 会明显下降。
2. Sequential `C+Q -> next G` 改变了内部机制，并出现更分布化的 flux 线索。
3. 不对 conductance logit 做 weight decay 后，模型仍能在较小 gamma 下使用 exchange。

### 仍未得到支持

1. **Opposing flow 比 same-direction flow 更有效。** V4 单 seed 最终准确率为正信号，但配对检验
   和训练 seed 证据不足，且 Co-current 的迭代纠错更强。
2. **Self-generated target hypothesis 的类别语义是主要收益来源。** Uniform/shuffled 几乎不降，
   当前 boundary 更像共同 reverse reservoir。
3. **多轮 relaxation 有额外价值。** Countercurrent 在 t=1 后已收敛，t=2/3 没有改善。
4. **相对普通 CNN 有精度/计算优势。** V4 CC 比参数匹配 CNN 低 0.42 pp，MACs 约为 4.74 倍。

## 7. 下一阶段建议

1. 保持 V4 结构和协议，先补 seed=1、2；若方向稳定，再扩到 5 seeds，报告 mean±std 和逐 seed
   `CC-Co` 差值。
2. 训练 T=1 的 CC/Co 成对控制，检查能否保留准确率并明显降低计算量。
3. 将 boundary 显式拆分为 `C_base + beta*C_class(p)`：公共 reservoir 与居中的 class-dependent
   分量分开记录、分开消融。这样可以判断类别 hypothesis 是否真正被使用。
4. 若增强 class 分量，可考虑低秩 spatial class prototype；不要同时改变通道、深度、T 和 loss。
5. 补 Forward-only Recurrent、Feedback-Fusion、从头训练的 Null/learned-Z 对照，排除额外计算、
   普通 feedback 和公共 reservoir 的解释。

在多个 seed 下稳定看到 `CC > Co`、真实 hypothesis 明显优于 uniform/shuffled、且 equal-compute
controls 不能解释收益之后，再把主实验扩展到 CIFAR-100。

## 8. 验证状态

- 46 项 unit tests 通过；
- V4 真实 CIFAR-10 单 batch forward/backward 通过；
- source 每轮重新注入、Q 非零、gamma 合法、所有状态和梯度有限；
- 测试覆盖 corrected C 进入相邻 G、CC/Co 镜像调度、旧 checkpoint 语义恢复及跨版本续训拒绝。
