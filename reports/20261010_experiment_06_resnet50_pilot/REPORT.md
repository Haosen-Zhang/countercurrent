# 第 6 次实验：ResNet-50 逆流晶格 Pilot（未完成）

- 时间：2026-10-10 18:37 – 22:06（Asia/Shanghai，提前终止）
- 数据集：CIFAR-10（45k/5k/10k）
- 设备：本机 M1 Pro / 10 核 / 16GB / MPS
- 状态：**未完成** —— 2/4 臂完成，1 臂部分完成，1 臂未开始

## 1. 实验目的

把逆流机制放到**标准 ResNet-50** 上，回答「countercurrent 是否优于 base CNN 或 co-current」。
此前所有逆流实验都基于自制的 812k 参数小骨干，对比缺乏说服力。

## 2. 架构设计（已实现，见 `countercurrent_nn/models/resnet_counterflow.py`）

以 ResNet-50 的 layer3 作为逆流晶格：block 0 保留为 stride-2 入口，blocks 1–5 变为 5 个
同形状（8×8×1024）可逆 bottleneck。反传用 `G = F⁻¹` 权重共享，不增加参数。

四条臂：

| 臂 | 说明 |
|---|---|
| `resnet50_plain` | 标准 ResNet-50（CIFAR stem），1 次前向 |
| `resnet50_lattice_control` | 同样的可逆 layer3，**关闭交换**，单次前向 |
| `resnet50_countercurrent` | C 在 depth L 注入，L→0 传播 |
| `resnet50_cocurrent` | 同求解器/边界/交换律，C 在 depth 0 注入，0→L |

参数公平性：唯一参数计数 23,897,624 vs plain 23,520,842，**+1.60%**。

## 3. 已完成的结果（3 epochs，seed 0）

| 臂 | params | MACs | compute | best val | **test** | test loss | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_plain` | 23,520,842 | 1,297,829,888 | 1.00× | 57.08% | **58.62%** | 1.2022 | 755 s |
| `resnet50_lattice_control` | 23,897,624 | 1,297,829,888 | 1.00× | 64.10% | **63.08%** | 1.0326 | 783 s |

### 3.1 可逆 layer3 在短 schedule 下高出 +4.46 pp

同算力下 `lattice_control` 显著优于 `plain`。**这不是逆流的功劳**：两者差别是把 layer3
换成 GroupNorm + SiLU + β=0.1 谱归一化的可逆残差块。epoch 1 时 plain 的 val loss 高达
5.03（train 1.96），BN 统计量在这个截断 warmup 下严重失准，而 GN 没有该问题。

**判断：大概率是短 schedule 假象**，200 epoch 下 BN 应追平。这是假设，**未经验证**。

### 3.2 协议偏差（重要）

配置中 `warmup_epochs: 5` 超过本次仅跑的 3 epochs，四条臂全程处于 warmup 斜坡
（lr 0.02→0.06）。横向比较公平，但**绝对精度远低于**正常 200-epoch 训练
（ResNet-50 on CIFAR-10 应约 93%）。这些数字不可作为 ResNet-50 的性能引用。

## 4. 部分结果：countercurrent（仅 1/3 epoch，已终止）

| 指标 | 值 |
|---|---:|
| epoch 1 train acc | 32.20% |
| epoch 1 val acc | 47.08% |
| train 耗时（45000 图） | 2673.7 s（44.6 min） |
| validation 耗时（5000 图） | 3029.9 s（50.5 min） |

同期 plain val 31.52%、control val 40.06%。**单 epoch、且 control 自身轨迹剧烈波动
（40.06 → 46.12 → 64.10），不构成证据。**

## 5. 关键发现：真实墙钟开销比 MACs 估计差一个量级

| 阶段 | plain | countercurrent | 倍数 |
|---|---:|---:|---:|
| train / epoch | 242 s | 2674 s | **11×** |
| validation | 7.8 s | 3030 s | **389×** |

基于 Conv/Linear MACs 的估计是 6.09×，**严重低估**。MACs 只统计算术量，不统计：

- `G=F⁻¹` 每次求逆需 4 次**顺序**残差求值（5 block × 2 sweep × 2 步），全是小算子；
- MPS 上小算子的 kernel 启动开销主导；
- 评估是**逐样本自适应早停**，Python 层循环，实测 1.65 图/秒。

按实测推算，coupled 臂完整 3 epochs + 最终测试评估约需 **6.4 小时/臂**，
两条臂约 13 小时，远超原先估计的 3 小时。**这是实验提前终止的直接原因。**

## 6. 结论边界

**可以支持**

1. ResNet-50 逆流架构实现正确：参数公平 +1.60%，112 项单元测试通过，训练管线在 MPS 可用。
2. 可逆 layer3（GN + 谱归一化 + β=0.1）在 3-epoch 截断 warmup 下比标准 BN layer3 高 4.46 pp（同算力）。

**不能支持**

1. **无法判定 countercurrent 是否优于 plain 或 co-current** —— coupled 臂未跑完。
2. 无法判定上述可逆层优势在完整 schedule 下是否保持。
3. 58.62% / 63.08% 不是可引用的 ResNet-50 性能（截断 warmup，仅 3 epochs）。

## 7. 复现与后续

产物目录仅含小文件（summary/metrics/config/profile/log），checkpoint 因体积未上传：

```
reports/20261010_experiment_06_resnet50_pilot/
  pilot_run.log            完整运行日志（含终止点）
  plain/                   summary.json metrics.jsonl config.yaml profile.json
  lattice_control/         同上
  countercurrent/          仅 1 epoch 的 metrics（部分）
  cocurrent/               仅 config（未开始训练）
```

重新运行：

```bash
cd /Users/haosen/Developer/countercurrent
# 前置：CIFAR-10 已在 dataset/（本机下载后 MD5 校验通过）
nohup ./run_resnet50_pilot.sh > /dev/null 2>&1 &
python -m countercurrent_nn.analysis.resnet_pilot_report
```

**再跑之前必须先解决的计算问题**：

1. 评估开销（占单 epoch 的 53%）：把 `eval_max_steps` 从 24 降到 6，或在 pilot 阶段
   用 `--limit-eval-batches` 缩小验证集；
2. 求逆开销：`inverse_steps` 从 4 进一步降低，或改用共享的独立反传网络（一次前向代替 4 次求逆）；
3. 两条 coupled 臂必须使用**相同**的评估设置以保持可比。
