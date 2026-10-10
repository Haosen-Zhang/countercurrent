# Countercurrent V5-C 后续验证流程

> 状态：Experiment 05 已判定 No-Go。V5-C CC/Co 的 solver 和 canonical exchange 正常，
> 但 class boundary 被 instance branch 绕过，且 CC 没有优于 Co。以下流程用于修复机制后再
> 决定是否进入 200 epochs 与 CIFAR-100。

## 1. 当前必须保留的结论

1. 不继续当前 V5-C 的 200-epoch 训练。
2. 不用更大数据集掩盖 class semantics collapse。
3. 保留 Softmax；当前证据指向分量尺度竞争，而非 CNN 与 Softmax 不兼容。
4. CC、Co 和 Raw control 必须继续保持相同 depth、参数量和最大 MAC。
5. 每次先跑 seed 0 的 20-epoch pilot，再决定长训练。

## 2. 下一版边界：V5-D guarded decomposed boundary

下一版只修改 boundary competition，不同时修改 solver、transport、canonical transform、
gamma 或 loss。

### 2.1 分量 RMS normalization

对 base、class、instance 分别做 per-sample RMS normalization，再乘显式系数：

```text
base_hat  = base / rms(base)
class_hat = class / rms(class)
inst_hat  = instance / rms(instance)
```

Class 分量在 uniform probability 时必须仍严格为零，不能将接近零的噪声放大。使用
Softmax confidence gate：

```text
p_centered = p - 1/K
confidence = ||p_centered||_2 / sqrt(1 - 1/K)   # [0, 1]
class_term = confidence * normalize_nonzero(class_map)
```

### 2.2 第一阶段固定 mechanism coefficients

首个 pilot 固定：

```text
lambda_base     = 0.15
lambda_class    = 1.0
lambda_instance = 0.1
```

它们使用 buffer，不参与优化。保留正 global gain，但 weight decay 为 0。这样可避免
`instance_scale: 0.1 -> 1.5` 以及 component/scale 的互相补偿。若 gate 通过，后续才测试
有界可学习系数。

### 2.3 保留的约束

- instance encoder 继续读取 `H_terminal.detach()`；
- centered Softmax 与 spatial class projector 保留；
- inner solve 中 boundary 固定；
- training 固定 8 inner steps；
- evaluation 逐样本最多 24 steps；
- loss 仍只用 final CE。

## 3. 实现前后的测试

必须新增或更新以下行为测试：

1. uniform probability 产生严格零 class term；
2. 非零 base/class/instance 在 normalization 后 RMS 符合定义；
3. instance effective coefficient 始终为 0.1；
4. confidence gate 在 uniform 时为 0、one-hot 时为 1；
5. instance reference 保持 stop-gradient；
6. CC/Co/Raw 的参数量和最大 MAC 严格相同；
7. 所有参数得到有限梯度；
8. toy BVP、全仓库测试和真实 CIFAR-10 one-batch sanity 均通过。

## 4. V5-D seed-0 pilot 顺序

### Step A：Countercurrent 20 epochs

只先跑 V5-D Countercurrent seed 0。逐 epoch 保存：

- final/initial equation residual 与 convergence rate；
- `R_base/R_class/R_instance`；
- confidence gate；
- raw 与 normalized component RMS；
- `abs(cos(D,u_H))`；
- initial/final iteration accuracy。

### Step B：冻结 checkpoint intervention

至少在完整 CIFAR-10 test set 比较：

1. real hypothesis；
2. uniform hypothesis；
3. shuffled hypothesis；
4. class-off；
5. instance-off；
6. base-only；
7. zero boundary。

### Step C：Countercurrent pilot gate

进入 Co-current pilot 前要求：

- test convergence rate ≥95%；
- final equation residual `<1e-3`；
- `R_class` 不下降超过一个数量级，且 final mean ≥0.05；
- real hypothesis 的 full-test accuracy 至少比 uniform/class-off 高 0.3 pp，并且 CE 更低；
- final mean/max `abs(cos(D,u_H)) < 0.95`。

0.3 pp 是 pilot go/no-go 阈值，不是论文显著性结论。

### Step D：严格匹配 Co-current 20 epochs

只有 Step C 通过才运行 Co-current。使用相同 seed、schedule、batch、参数、MAC 和干预。
若 CC 与 Co 差异小于 0.2 pp，则视为尚无 topology signal；不进入长训练。

### Step E：compute/parameter-matched Raw 20 epochs

CC class gate 与 topology gate 都通过后，再运行 Raw Exchange control，分离 persistent solver、
canonical alignment 和 opposing direction 的贡献。

## 5. 长训练与多 seed

只有 CC、Co、Raw 三个 seed-0 pilots 全部通过机制检查后，才从头运行 200 epochs。Pilot
checkpoint 不 resume 到正式训练，因为 cosine schedule 不同。

正式顺序：

1. V5-D Countercurrent seed 0，200 epochs；
2. V5-D Co-current seed 0，200 epochs；
3. V5-D Raw Countercurrent seed 0，200 epochs；
4. 冻结 checkpoint 全测试集 interventions；
5. 若方向和语义 gate 仍成立，补 seeds 1、2；
6. 报告 mean ± std、逐 seed paired difference、MAC、latency、memory 和 solve-step distribution。

## 6. 进入 CIFAR-100 的条件

同时满足以下条件后才切换 CIFAR-100：

1. ≥95% test samples 在 24 steps 内收敛；
2. real hypothesis 稳定优于 uniform、shuffled 和 class-off；
3. 至少 3 个 CIFAR-10 seeds 上 `CC > Co` 方向一致；
4. exchange 未退化为 H scaling；
5. Raw/Canonical/Co controls 的预算匹配已验证。

若 V5-D 在强制 class semantics 后显著降低准确率，应该把它视为反证并记录，而不是重新开放
instance shortcut 或直接扩大数据集。
