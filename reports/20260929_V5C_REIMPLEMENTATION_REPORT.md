# Countercurrent V5-C 重实现报告（2026-09-29）

本次修改以 `COUNTERCURRENT_V5C_DETAILED_PLAN.md` 为实现规范。旧
`countercurrent_nn/models/bvp_lattice.py` 保留，V5-C 作为新的模型族注册，因而不会改变
旧 checkpoint 的语义。本阶段没有启动 epoch training。

## Architecture diff

| 旧 BVP / V4 | V5-C |
|---|---|
| 每次 sweep 从 target boundary 重新生成整条 C | C 仅在首次 inner solve 初始化，之后跨 inner step 持续传播 |
| inner step 内可重新计算右边界 | outer step 只构造一次 boundary，整个 inner solve 固定并逐步 clamp |
| 顺序 reconciliation，更新顺序可能引入偏置 | Jacobi synchronous map，所有 cell 只读取上一步 H/C |
| 直接在 H/C 原坐标做 `H-C` | 分别用正交 `A_H/A_C` 映射到 canonical space，再计算同一个 paired flux |
| target boundary 的类别空间表达较弱 | `base + centered spatial class + stop-gradient instance` 三分量边界 |
| 训练和评估可共享 adaptive solver | `unroll_train` 固定 8 步 full BPTT；`solve_eval` 才逐样本提前停止 |
| 主要观察 step change | 用未阻尼 fixed-point map 同时计算 H/C interior equation residual |
| Co-current 对照可能产生数据相关计算量差异 | Co/Counter 使用完全相同模块和最大 inner-step MAC 预算，只改变 C 的传播方向 |

V5-C 的单个 cell 为：

```text
h_bar = F_l(H_l^s)
c_bar = G_l(C_neighbor^s)
u_h   = A_H(h_bar)
u_c   = A_C(c_bar)
D     = u_h - u_c
Q     = gamma * D,  0 < gamma < 0.49
H*    = A_H^T(u_h - Q)
C*    = A_C^T(u_c + Q)
```

随后使用独立的 solver damping `rho=0.5` 更新状态，并重新 clamp
`H_0=Stem(x)` 和 C 的 target boundary。Canonical 子步骤满足
`u_h_new + u_c_new = u_h + u_c`。

## 新实现

- `countercurrent_nn/models/canonical_exchange.py`：正交通道变换、canonical discrepancy、
  channel-wise positive conductance 和 paired exchange。
- `countercurrent_nn/models/decomposed_boundary.py`：base、中心化类别空间图、stop-gradient
  instance 与正 global gain；uniform class probability 的类别项严格为零。
- `countercurrent_nn/models/persistent_bvp.py`：固定双边界、persistent H/C、同步 Jacobi、
  固定步训练、逐样本自适应评估，以及完整机制诊断。
- `countercurrent_nn/configs/cifar10_v5a_countercurrent_raw.yaml`：Experiment A，persistent
  fixed-boundary + raw exchange。
- `countercurrent_nn/configs/cifar10_v5c_countercurrent.yaml`：Experiment B，V5-C 主模型。
- `countercurrent_nn/configs/cifar10_v5c_cocurrent.yaml`：Experiment C，严格镜像 Co-current 对照。
- `countercurrent_nn/v5c_sanity_check.py`：单批次 forward、backward、optimizer update、
  boundary clamp、adaptive solve、reverse-off 与 profiling 检查。

## 验证结果

### Unit tests

```text
V5-C 专项与 toy BVP：22 passed
全仓库：88 passed
compileall：passed
```

专项测试覆盖 C persistence、boundary call count、两端 clamp、训练固定步数、评估逐样本
停止、equation/step residual 分离、canonical antisymmetry、正交性、双向梯度影响、uniform
class、spatial class、instance stop-gradient、边界 RMS 标定、无 GT leakage、全参数有限梯度，
以及 Co/Counter 参数、state dict shape 和 MAC 的严格一致性。

线性 toy BVP 通过以下 gate：迭代解逼近直接线性方程解；damping 改变收敛速度但不改变
fixed point；adaptive early-stop 与固定长迭代一致。

### 真实 CIFAR-10 单批次 sanity check

使用 GPU 2、4 个真实 CIFAR-10 training samples，只执行一次 AdamW update：

| 检查项 | 结果 |
|---|---:|
| loss before / after one update | 2.128571 / 1.890371 |
| fixed train inner steps | 8 |
| eval steps per sample | 14, 14, 14, 14 |
| eval equation residual | 0.000802–0.000809 |
| converged within 24 steps | 4 / 4 |
| source / target boundary RMS | 0.637 / 0.460 |
| reverse-off max logit difference | 0.105024 |
| finite states and gradients | passed |
| source/target clamp | passed |
| boundary call count | passed |

完整数值见 `reports/v5c_one_batch_sanity.json`。这里的 loss 变化和初始 batch accuracy
只证明链路能训练，不能视为模型精度结果。

### 公平性与成本

按 batch=1、24 个最大 inner steps 统计 Conv/Linear MAC：

| 配置 | 参数 | MACs |
|---|---:|---:|
| V5-C Countercurrent | 812,270 | 1,010,318,976 |
| V5-C Co-current | 812,270 | 1,010,318,976 |
| V5-A raw Countercurrent（最初未匹配版本） | 779,502 | 959,987,328 |

Co/Counter 的参数和最大计算预算完全一致。实际 eval latency仍受逐样本提前停止步数影响，
应在正式实验中单独报告 solve-step distribution 和 latency。

## 当前结论与下一步 gate

实现层面的 V5-C 条件已经满足，toy BVP 和单批次数值稳定性也已通过。当前结果不能证明
Countercurrent 的分类优势，因为尚未训练模型，也尚未做真实/均匀/打乱 hypothesis 干预。

下一步仍应留在 CIFAR-10，按相同 seed 依次训练 V5-A raw、V5-C Countercurrent、V5-C
Co-current。seed 0 先检查：测试集 24 步内收敛率、class boundary 干预、
`cos(D,u_H)` 是否退化为 scaling，以及 CC/Co 的方向差异。通过这些 gate 后再补 3 个 seeds；
在 CC 稳定优于 Co 且 target hypothesis 有明确贡献之前，不进入 CIFAR-100。

## 长训练前 small patch

2026-09-29 的后续 patch 将 `base_scale`、`class_scale`、`instance_scale` 和
`log_global_gain` 与 conductance 一起放入 weight decay 0 参数组。Raw control 使用两条
严格正交的 encode/decode round-trip；round-trip 数学上为 identity，所以仍严格计算
`D=H-C`，同时补齐 Canonical 模型的参数和 Linear MAC。修改后 V5-A Raw、V5-C
Countercurrent 和 V5-C Co-current 均为 812,270 参数、1,010,318,976 MAC。

V5 三份配置还会在每个 epoch 的 `metrics.jsonl` 中记录紧凑机制指标：最终 equation
residual、收敛率、`R_class` 与最终 `|cos(D,u_H)|`。终端也会打印 `req`、`Rclass` 和
`cosD`，用于 10–20 epoch diagnostic pilot 的 go/no-go 判断。

Patch 后全仓库 92 项测试通过。额外的 1 epoch / 1 batch synthetic smoke 成功写出机制日志：
`req=8.045e-4`、`Rclass=0.1372`、`cosD=0.7776`。这些 synthetic 数值只验证日志链路，
不作为机制或精度结论。
