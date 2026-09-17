# Countercurrent CNN Proof of Concept

本仓库实现两份最新版研究规范中的第一阶段 CNN 实验：用 source evidence 与模型自己生成的
target hypothesis 作为两端 boundary，在推理阶段进行 counterflow paired exchange。

- [核心思想](COUNTERCURRENT_CORE_IDEA.md)
- [实验规范](COUNTERCURRENT_CNN_POC_EXPERIMENT.md)
- [重实现要求](COUNTERCURRENT_REIMPLEMENT_PROMPT.md)
- [本次审计、architecture diff 与验证报告](reports/COUNTERCURRENT_REIMPLEMENTATION_REPORT.md)

2026-09-14：主模型已替换为 proposal → paired reconciliation → corrected forward resweep。
旧代码中的 `C + Q` 只写入 diagnostics；新版把该 corrected C 作为 resweep 的实际输入。
旧模型与配置保存在 `countercurrent_nn/legacy/old_wrong_countercurrent/`，不进入 active registry。

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

主模型不接收 label/GT，GT 只进入 task loss（v2 为最终 CE，v3 可加初始 CE）；最终分类头只读取 `H_L`。这里的
`-Q/+Q` 只是 exchange substep 的 zero-sum 约束，不代表整个网络能量守恒。

每轮先从 `stem(x)` 生成 forward proposal，并从上一轮 soft hypothesis 生成 reverse proposal。
第一阶段 paired exchange 的 `C + Q` 随后被 corrected forward resweep 逐层读取；resweep
再次使用同一个逐通道 conductance 做 paired exchange。两阶段各自使用一份 `Q` 更新 H/C，
并分别记录 `proposal_D/proposal_Q` 与 `D/Q`。这是实验规范 §15 的有限 sweep 方案，
不声称每轮已经解出同步 lattice 的 equilibrium。

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
243,022,336 Conv/Linear MACs；两者完全相等。Forward recurrent 为 668,362 参数、
243,010,048 MACs，控制“多算几轮”本身；8-block 单流模型为 668,362 参数，作为近似参数匹配对照。
以上均按 batch=1；`2 × MACs` 为所统计 Conv/Linear 的近似 FLOPs，不包含 normalization、
activation、prototype matmul 与 elementwise exchange。新 schedule 的配置使用 `_v2_seed0`
输出目录，避免覆盖旧 schedule 的结果。旧 checkpoint 应通过 legacy 模型读取。

## 测试

```bash
conda run -n 4dflow python -m unittest discover -s tests -v
conda run -n 4dflow python -m countercurrent_nn.sanity_check --batch-size 8
```

测试覆盖 soft boundary、无 GT leakage、每轮 source clamp、`t=0` 纯 forward、conductance
范围、paired-exchange 代数、双向局部影响、reverse-off、所有 state shape，以及 CC/Co 的参数量和
MAC 严格一致性。

`sanity_check` 默认使用明确标注的 synthetic input，只做一个 batch 的 forward/backward、
一次 AdamW update、reverse-off 和 profiling；不启动 epoch training。
若已有 CIFAR-10 数据，可传 `--data-root /path/to/data`（不会下载数据）。
结果写到 `reports/one_batch_sanity.json`，完整初始 tensor diagnostics 写到同名 `.pt`。
GT Oracle 仅通过 `countercurrent_nn.analysis.oracle_boundary.GTOracleCountercurrent` 显式分析，
不进入主模型 registry，也不属于正常 test accuracy。

## 训练与评估

先跑完整链路的合成数据 smoke test：

```bash
conda run -n 4dflow python -m countercurrent_nn.train \
  --config countercurrent_nn/configs/cifar10_countercurrent_selfboundary.yaml \
  --debug --epochs 1 --synthetic-data \
  --limit-train-batches 1 --limit-eval-batches 1 \
  --output-dir /tmp/countercurrent_smoke
```

三组 seed=0 的 CIFAR-10 v2 实验已完成，结果见
[三组实验分析](reports/cifar10_threeway_20260916/REPORT.md)。新方案先进行单批次检查，
需要短程 debug 时可使用 `--debug`（默认 3 epochs）。

CIFAR-10 主实验（后续阶段）：

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

合成数据 smoke test 只验证执行链路，其指标不作为 CIFAR 科研结果。

## 单机四卡训练

训练入口支持 `torchrun` DDP：每进程使用一张 GPU，训练数据通过 DistributedSampler
分片并在每个 epoch 重新 shuffle。训练指标在各进程间汇总；验证、最终测试、日志和
checkpoint 只由 rank 0 处理。保存的模型权重不带 `module.` 前缀，可直接用于现有评估命令。
验证/测试读取完整 split，不使用可能补齐重复样本的 distributed sampler。

`--batch-size` 表示全局 batch；四卡使用 128 时，每卡是 32。学习率不会按卡数自动放大。
`--num-workers` 是每个进程的 DataLoader worker 数量。以下两项实验依次运行：

```bash
conda activate 4dflow
cd /localhome/zhanghs/Countercurrent

# 普通 CNN：参数量匹配版（L=8，668,362 参数）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29501 \
  -m countercurrent_nn.train \
  --config countercurrent_nn/configs/cifar10_single_parammatched.yaml \
  --data-root ./dataset --no-download --device cuda \
  --batch-size 128 --epochs 200 --seed 0 \
  --output-dir countercurrent_nn/results/cifar10_single_parammatched_4gpu_seed0

# Countercurrent：self-generated target boundary（L=4，T=3，673,418 参数）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29502 \
  -m countercurrent_nn.train \
  --config countercurrent_nn/configs/cifar10_countercurrent_selfboundary.yaml \
  --data-root ./dataset --no-download --device cuda \
  --batch-size 128 --epochs 200 --seed 0 \
  --output-dir countercurrent_nn/results/cifar10_countercurrent_v2_4gpu_seed0
```

先做短程检查时，在对应命令中把 `--epochs 200` 改为 `--debug --epochs 3`，
并为 debug 使用独立输出目录。默认正式配置为 SGD、lr=0.1、momentum=0.9、
weight decay=5e-4、5 epochs warmup + cosine decay。两模型读取同一 45k/5k/10k split。

## v3 与训练消融（2026-09-16）

继续 CIFAR-10，复用原 v2 seed=0 作为对照。以下选项仅改变训练，不改变 v2 的推理拓扑、
参数量、MACs、L=4 或 T=3；旧配置省略新选项时仍使用原有训练目标和参数分组。

| 配置后缀 | `conductance_weight_decay` | `initial_loss_weight` |
|---|---:|---:|
| `gamma_nodecay` | 0 | 0 |
| `initial_aux` | 默认随普通参数：0.0005 | 0.2 |
| `v3` | 0 | 0.2 |

每个后缀都有 `cifar10_cocurrent_*.yaml` 与 `cifar10_countercurrent_*.yaml`。v3 仅对
`ChannelwiseConductance.logit_gamma` 取消衰减，其余参数仍为 0.0005。
辅助目标为 `CE(logits_T,y) + 0.2*CE(logits_0,y)`，不将标签传入 forward。
初始与最终 logits 在同一次 DDP forward 中返回；不额外运行一次模型。
训练日志的 `loss` 为加权总目标，`final_loss` 为最终 CE，启用辅助监督时另存
`initial_loss` 和 `initial_accuracy`；验证、选取 best checkpoint 和测试仍只使用最终输出。

下面的命令依次运行六个实验，每个实验使用四张 GPU。先激活已安装依赖的环境。
默认 200 epochs、全局 batch=128、每卡 32，原 v2 不需重跑。

```bash
conda activate 4dflow
cd /localhome/zhanghs/Countercurrent
export CUDA_VISIBLE_DEVICES=0,1,2,3

for variant in gamma_nodecay initial_aux v3; do
  for model in cocurrent countercurrent; do
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port=29517 \
      -m countercurrent_nn.train \
      --config "countercurrent_nn/configs/cifar10_${model}_${variant}.yaml" \
      --data-root ./dataset --no-download --device cuda \
      --batch-size 128 --epochs 200 --seed 0 \
      --output-dir "countercurrent_nn/results/cifar10_${model}_${variant}_4gpu_seed0" \
      || exit 1
  done
done
```

只训练组合改进 v3 时，将外层循环改为 `for variant in v3; do`；完整消融仍建议运行三个后缀。
改变 seed 时同时改变命令中的 `--seed` 和输出目录的 `seed0`。数据划分的 `split_seed=0`
保持不变。所有新设置从头训练，不从旧 v2 checkpoint 接着训练；`--resume` 只用于同一设置的中断续训。
入口拒绝无 `--resume` 覆盖已有 checkpoint/metrics，并检查续训时两个新选项与 checkpoint 一致。

先按验证集筛查，再为选中方案和 v2 对照补 seeds=1、2。确认后再扩展 CIFAR-100；本轮同时
换数据集会失去现成对照，不利于判断性能变化来自哪一项改动。

四进程梯度与日志汇总检查（CPU/Gloo，两个小批次，不是正式训练）：

```bash
OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29517 \
  -m tests.distributed_ablation_smoke
```

## v4 sequential reverse reconciliation（2026-09-17）

v4 强化 reverse stream 内部的证据传播。v2 先独立计算完整的 G proposal，再对所有层交换；
v4 则在每个 reverse cell 完成 transport 后立刻做 paired exchange，并把修正后的 `C + Q`
作为相邻 G block 的输入：

```text
C[L] = target boundary
for l = L-1 ... 0:
    C_bar[l] = G[l](C[l+1])
    Q[l] = gamma[l] * (H_prop[l+1] - C_bar[l])
    H_rec[l+1] = H_prop[l+1] - Q[l]
    C[l] = C_bar[l] + Q[l]  # live inlet to G[l-1]
```

之后仍执行 corrected forward resweep，每轮仍重新注入 `stem(x)`，boundary 仍来自模型自身的
soft hypothesis，局部交换仍严格使用同一个 Q 的 `-Q/+Q`。Co-current 使用同样的逐 cell
调度，只把 C 的传播顺序改为 `0 -> L`。

新配置为 `cifar10_countercurrent_v4_sequential.yaml` 和
`cifar10_cocurrent_v4_sequential.yaml`，显式设置 `inference_version: 3`。它们使用最终 CE、
不使用 v3 的初始 CE，并只对 conductance logit 取消 weight decay。已有配置显式保留
`inference_version: 2`；旧 checkpoint 中的版本 buffer 也会恢复原调度，因此旧实验结果不会
被新代码重新解释。训练 summary 和 diagnostics 会记录 inference version。
训练入口也拒绝用不同 inference version 的 checkpoint 续训 v4。

46 项单元测试通过，覆盖 corrected C 进入相邻 G、两种方向的镜像调度、旧 checkpoint 调度
恢复、paired flux、梯度、无 GT leakage 和所有原有回归测试。真实 CIFAR-10 单批次报告为
`reports/v4_sequential_one_batch_sanity.json`。
