# Countercurrent CNN Proof-of-Concept：详细实现与初始实验规范

> **目标**：直接交给 Codex 实现第一阶段 CNN proof-of-concept。  
> **首要科学问题**：当 source evidence 与 self-generated target hypothesis 同时作为两端 boundary 时，opposing-flow 的 layerwise paired exchange 是否比 same-direction coflow / forward-only recurrent refinement 更有效？  
>
> **主模型原则**：
>
> \[
> \boxed{
> H_0^t=E(x)\quad \forall t
> }
> \]
>
> 每个 inference iteration 都重新注入同一个输入 evidence。
>
> target-side boundary 由上一轮 forward terminal state产生：
>
> \[
> \boxed{
> C_L^{t+1}=B(P(H_L^t))
> }
> \]
>
> GT 只进入最终 Cross-Entropy loss，不进入 reverse stream。

---

# 1. 第一阶段必须回答的三个问题

## Q1：Countercurrent topology 是否真的优于 Co-current？

在相同：

- source input；
- target hypothesis；
- channel；
- depth；
- transport blocks；
- exchange operator；
- iterations；
- classifier；
- params；
- 近似 FLOPs；

下：

\[
\boxed{
Acc_{\rm CC}
>
Acc_{\rm Co}
\;?
}
\]

---

## Q2：Reverse pathway 是否真的参与 inference？

训练完成后测试：

\[
Q_l=0
\]

或关闭 reverse stream。

如果：

\[
Acc_{\rm full}
>
Acc_{\rm reverse-off},
\]

说明 reverse path 是 computation 的必要部分，而不只是 training regularizer。

---

## Q3：Iterative hypothesis feedback 是纠错还是 confirmation bias？

统计：

- initial wrong → final correct；
- initial wrong → final more-confident wrong；
- iteration-wise accuracy；
- entropy / confidence trajectory。

---

# 2. 数据集

主：

## CIFAR-10

- 50,000 train；
- 10,000 test；
- 10 classes；
- 32×32 RGB。

正式确认可使用：

- 45k train；
- 5k validation；
- 10k untouched test。

第二阶段：

## CIFAR-100

保持 architecture 和 training setting 尽量不变，只改：

\[
10\rightarrow100
\]

classes。

---

# 3. 第一版整体网络

输入：

\[
x\in\mathbb R^{B\times3\times32\times32}.
\]

Stem：

\[
E(x)\rightarrow H_0\in\mathbb R^{B\times64\times8\times8}.
\]

lattice 内所有 H/C state 固定同 shape：

\[
H_l,C_l
\in
\mathbb R^{B\times64\times8\times8}.
\]

depth：

\[
L=4.
\]

初始推荐 inference refinement steps：

\[
T=3.
\]

注意这里区分：

- depth \(L=4\)；
- outer hypothesis-refinement iterations \(T=3\)。

不要混成一个变量。

---

# 4. 为什么 lattice 内必须同 shape

不要第一版设计：

```text
H: 32 -> 16 -> 8
C: 8 -> 16 -> 32
```

否则会和 encoder-decoder / U-Net / upsampling confound。

第一版：

\[
[B,64,8,8]
\]

在所有 depth 保持不变。

最终 scale-up 再做 stage-wise ResNet/ConvNeXt。

---

# 5. Stem

推荐：

```python
Stem(
    Conv3x3(3, 64, stride=1),
    GroupNorm(8, 64),
    SiLU(),

    Conv3x3(64, 64, stride=2),  # 32 -> 16
    GroupNorm(8, 64),
    SiLU(),

    Conv3x3(64, 64, stride=2),  # 16 -> 8
    GroupNorm(8, 64),
    SiLU(),
)
```

得到：

```python
H0 = stem(x)  # [B,64,8,8]
```

推荐 GroupNorm 而不是 BatchNorm，避免多 iteration distribution 混合 running statistics。

---

# 6. Transport Block

推荐：

```python
class TransportBlock(nn.Module):
    def __init__(self, channels=64, residual_scale=0.1):
        ...
```

结构：

```text
x
|-------------------------------+
|                               |
GN -> SiLU -> Conv3x3            |
              -> GN -> SiLU -> Conv3x3
                                |
                           * residual_scale
                                |
                                +
                                |
                                y
```

公式：

\[
T_l(x)=x+\beta f_l(x),
\qquad
\beta=0.1.
\]

主版本：

- Forward transport：`F[l]`
- Reverse transport：`G[l]`
- architecture 相同；
- weights 独立；
- Countercurrent / Co-current 参数数量完全一致。

后续可做 mirror/shared-weight ablation，但不是首轮硬要求。

---

# 7. 第 0 轮：纯 forward 初始化

每个样本首先只跑一次 forward：

\[
H_0^0=E(x).
\]

\[
H_{l+1}^0=F_l(H_l^0).
\]

得到：

\[
H_L^0.
\]

初始 logits：

\[
s^0=Head(GAP(H_L^0)).
\]

概率：

\[
p^0=\operatorname{softmax}(s^0).
\]

记录：

- `initial_logits`
- `initial_acc`
- `initial_entropy`

第 0 轮没有 reverse exchange。

---

# 8. Self-generated Target Boundary

## 8.1 Class prototype bank

学习：

\[
Z=
[z_1,\dots,z_K],
\qquad
z_k\in\mathbb R^{d_z}.
\]

推荐：

\[
d_z=64.
\]

PyTorch：

```python
self.class_prototypes = nn.Parameter(
    torch.randn(num_classes, 64) * 0.02
)
```

---

## 8.2 Soft hypothesis

当前概率：

```python
p = softmax(logits, dim=-1)  # [B,K]
```

构造：

\[
z_{\rm hyp}
=
\sum_k p_k z_k.
\]

代码：

```python
z_hyp = p @ self.class_prototypes  # [B,64]
```

不要使用：

```python
argmax(p)
```

作为主版本。

---

## 8.3 Boundary projector

把：

\[
[B,64]
\]

映射为：

\[
[B,64,8,8].
\]

最简单：

```python
boundary = Linear(64, 64)(z_hyp)
boundary = boundary[:, :, None, None].expand(-1, -1, 8, 8)
```

或者：

```python
Linear(64, 64*8*8)
```

但后者参数更多。

第一版推荐 channel vector broadcast。

于是：

\[
\boxed{
C_L^{t}=B(z_{\rm hyp}^{t-1})
}
\]

---

# 9. 每个 outer iteration 都必须重新注入 x

这是硬约束。

每个：

\[
t=1,\dots,T
\]

都重新：

```python
H[0] = stem(x)
```

即：

\[
\boxed{
H_0^t=E(x).
}
\]

不能：

```python
H0 = old_H0
```

然后脱离原图。

source boundary 必须始终是固定 evidence anchor。

---

# 10. Counterflow Exchange Operator

第 \(l\) 个 cell 输入：

\[
H_l
\]

以及：

\[
C_{l+1}.
\]

先 transport：

\[
\tilde H_{l+1}
=
F_l(H_l)
\]

\[
\tilde C_l
=
G_l(C_{l+1}).
\]

然后：

\[
\boxed{
D_l=
\tilde H_{l+1}-\tilde C_l
}
\]

---

# 11. 第一版 exchange 不再使用 arbitrary 1×1 Conv

主版本推荐：

\[
\boxed{
Q_l
=
\gamma_l\odot D_l
}
\]

其中：

\[
0<\gamma_{l,c}<0.5.
\]

参数化：

\[
\gamma_l
=
0.49\sigma(a_l).
\]

PyTorch：

```python
class ChannelwiseConductance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.logit_gamma = nn.Parameter(torch.full((1, channels, 1, 1), -2.2))

    def forward(self, d):
        gamma = 0.49 * torch.sigmoid(self.logit_gamma)
        return gamma * d
```

初始：

\[
\sigma(-2.2)\approx0.10
\]

所以：

\[
\gamma\approx0.049.
\]

---

# 12. paired exchange

必须：

\[
\boxed{
H_{l+1}
=
\tilde H_{l+1}-Q_l
}
\]

\[
\boxed{
C_l
=
\tilde C_l+Q_l.
}
\]

主版本不要分别学习：

\[
Q_H,Q_C.
\]

这保证 exchange substep 是：

\[
\Delta H=-Q,\qquad \Delta C=+Q.
\]

---

# 13. 一个 Countercurrent Cell

```python
def cc_cell(h_in, c_in, F, G, conductance):
    h_bar = F(h_in)
    c_bar = G(c_in)

    d = h_bar - c_bar
    q = conductance(d)

    h_out = h_bar - q
    c_out = c_bar + q

    return h_out, c_out, d, q
```

注意：

```python
h_in  == H[l]
c_in  == C[l+1]

h_out == H[l+1]
c_out == C[l]
```

---

# 14. Outer iteration 内的 Countercurrent computation

这里不建议再次引入复杂的 inner-DEQ。

第一版每个 outer iteration 做：

1. 从左向右生成 temporary forward transported states；
2. 从右向左生成 temporary reverse transported states；
3. 对对应 depth 做 paired exchange；
4. 得到本轮 H/C；
5. 从新的 \(H_L\) 得到下一轮 hypothesis。

为了避免实现成 encoder-decoder，需要注意：

> reverse state 不是只在 forward 完成后用于最终 fusion；它必须在同一 outer iteration 中改变各层 forward state。

---

# 15. 推荐第一版实现：two-sweep coupled update

为了工程简单，第一版使用 two-sweep，而不是 DEQ/Jacobi lattice。

## Step A：forward transport proposal

每轮重新：

```python
H_prop[0] = stem(x)
for l in range(L):
    H_prop[l+1] = F[l](H_prop[l])
```

---

## Step B：reverse transport proposal

boundary：

```python
C_prop[L] = make_boundary(prev_logits)
```

然后：

```python
for l in reversed(range(L)):
    C_prop[l] = G[l](C_prop[l+1])
```

---

## Step C：layerwise paired reconciliation

对每个 cell：

\[
D_l=
H_{\rm prop,l+1}
-
C_{\rm prop,l}
\]

\[
Q_l=\gamma_l\odot D_l.
\]

更新：

\[
H_{l+1}
=
H_{\rm prop,l+1}-Q_l
\]

\[
C_l
=
C_{\rm prop,l}+Q_l.
\]

然后为了让 corrected H 真正影响下一层，推荐进行一次 **corrected forward resweep**：

```python
H_new[0] = stem(x)
for l in range(L):
    h_bar = F[l](H_new[l])
    q = gamma[l] * (h_bar - C[l])
    H_new[l+1] = h_bar - q
```

最终：

```python
logits = head(GAP(H_new[L]))
```

这比只在 layer proposal 后一次性修改 H 更能体现 reverse 对 forward computation 的真实影响。

---

# 16. 更严格的版本：synchronous lattice（第二阶段）

如果 two-sweep 有正信号，再实现：

\[
H_l^t,C_l^t
\]

的 Jacobi-style synchronous update。

所有 cell 在 iteration \(r+1\) 只读取 \(r\) 的状态。

这更接近 two-boundary coupled field，但工程更复杂。

第一轮不要求直接 DEQ。

---

# 17. 完整 inference 伪代码

```python
def forward(self, x, return_diag=False):
    source = self.stem(x)

    # iteration 0: pure forward
    H = [None] * (L + 1)
    H[0] = source
    for l in range(L):
        H[l+1] = self.F[l](H[l])

    logits = self.head(gap(H[L]))

    diagnostics = []

    for t in range(self.refine_steps):

        # --- fixed source boundary ---
        source = self.stem(x)

        # --- endogenous target boundary from previous H_L ---
        p = torch.softmax(logits, dim=-1)
        z_hyp = p @ self.class_prototypes
        cL = self.boundary_projector(z_hyp)
        cL = cL[:, :, None, None].expand(-1, -1, 8, 8)

        # --- reverse proposal ---
        C = [None] * (L + 1)
        C[L] = cL
        for l in reversed(range(L)):
            C[l] = self.G[l](C[l+1])

        # --- corrected forward pass, x is injected again ---
        H_new = [None] * (L + 1)
        H_new[0] = source

        q_list = []
        d_list = []

        for l in range(L):
            h_bar = self.F[l](H_new[l])

            # use locally corresponding reverse state
            c_ref = C[l]

            d = h_bar - c_ref
            q = self.exchange[l](d)

            h_new = h_bar - q

            # optionally paired-correct local reverse state for diagnostics/state
            C[l] = c_ref + q

            H_new[l+1] = h_new

            d_list.append(d)
            q_list.append(q)

        H = H_new
        logits = self.head(gap(H[L]))

        if return_diag:
            diagnostics.append({
                "logits": logits,
                "q": q_list,
                "d": d_list,
                "H": H,
                "C": C,
            })

    return logits if not return_diag else (logits, diagnostics)
```

注意：

- 每轮都执行 `source = stem(x)`；
- reverse boundary 来自上一轮 logits；
- reverse state 逐层参与 corrected forward；
- 不把 GT 输入 reverse；
- 不在最终 concat `H_L` 与 `C_0`。

---

# 18. Co-current control

使用完全相同：

- forward network；
- target prototypes；
- boundary projector；
- exchange conductance；
- refinement steps。

唯一主要区别：

Countercurrent：

\[
C_L=B(z)
\]

并：

\[
C:L\rightarrow0.
\]

Co-current：

\[
C_0=B(z)
\]

并：

\[
C:0\rightarrow L.
\]

代码概念：

```python
C[0] = c_boundary
for l in range(L):
    C[l+1] = G[l](C[l])
```

然后 forward 第 \(l\) 个位置使用 co-directional 对应 state。

Co-current 与 Countercurrent 必须参数量相同。

---

# 19. Null-boundary ablation

保留：

\[
C_L=0
\]

但定位改为：

### Null Counterflow

用途：

> 检查纯 opposing topology 在没有 target information 时是否仍有价值。

不要把它当 conceptual main model。

---

# 20. Learned-Z ablation

定义：

```python
self.c_boundary = nn.Parameter(
    torch.zeros(1,64,8,8)
)
```

所有样本共享：

\[
C_L=Z.
\]

比较：

- null；
- learned Z；
- self-generated target boundary。

---

# 21. GT Oracle ablation

只做分析：

\[
C_L=B(y_{\rm GT}).
\]

用途：

- oracle upper bound；
- 对比 self-inferred boundary；
- 估计 target-boundary inference 是否是瓶颈。

正式 test accuracy 不能把它和正常 inference 结果混为一谈。

---

# 22. Baselines

至少：

## A. SingleStream-FeedForward

普通：

```text
Stem -> F0 -> F1 -> F2 -> F3 -> Head
```

---

## B. Forward-Only Recurrent Refinement

每轮都重新注入 x，但没有 reverse：

\[
H^{t+1}=F(H^t,x)
\]

用于控制：

> 多算几次本身是否有用。

---

## C. Feedback-Fusion baseline

reverse/top-down state存在，但使用简单：

\[
H'=H+\phi(C)
\]

没有 paired flux。

用于证明：

> paired exchange 比普通 feedback 更重要。

---

## D. CCL-style training-only reverse

如果实现方便，可参考 CCL：

- reverse 只训练使用；
- test 关闭。

用于证明：

> 我们的 reverse 是 inference computation，而不是 training-only auxiliary branch。

---

## E. Co-current

最重要的 topology baseline。

---

## F. Countercurrent

主模型。

---

# 23. Training loss

第一版主模型：

\[
\boxed{
L=L_{\rm CE}(\hat y^T,y)
}
\]

可选加 intermediate deep supervision：

\[
L
=
L_{\rm CE}(p^T,y)
+
\lambda_{\rm iter}
\sum_{t<T}
L_{\rm CE}(p^t,y)
\]

但首轮建议：

\[
\lambda_{\rm iter}=0
\]

先保持最简。

不要第一轮加入 local exchange energy loss。

---

# 24. Local exchange energy：只记录，第二阶段再训练

定义：

\[
E_l^t
=
\frac12
(D_l^t)^\top
K_l
D_l^t.
\]

第一轮：

> 只作为 diagnostic，不作为 loss。

因为直接最小化所有：

\[
E_l
\]

可能导致 feature collapse 或过快消灭 discrepancy。

第二阶段才考虑：

\[
L=L_{\rm task}+\lambda_E\sum_lE_l.
\]

---

# 25. Flux-driven weight update：不要第一版实现成主优化器

理论：

\[
D=UH-VC
\]

\[
E=\frac12D^\top KD
\]

有：

\[
\Delta U=-\eta QH^\top
\]

\[
\Delta V=+\eta QC^\top.
\]

这是后续高级研究方向。

第一版：

\[
\boxed{\text{正常 end-to-end BP}}
\]

训练所有参数。

否则难以区分：

- architecture failure；
- local learning rule failure。

---

# 26. Augmentation

第一版：

```python
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(mean, std)
```

暂时不要：

- MixUp；
- CutMix；
- RandAugment；
- AutoAugment；
- heavy label smoothing。

---

# 27. Optimizer

Debug：

- AdamW；
- lr = 3e-4；
- wd = 1e-4；
- batch = 128；
- 20–30 epochs；
- seed 0。

Main：

- SGD；
- momentum = 0.9；
- weight decay = 5e-4；
- lr = 0.1；
- batch = 128；
- epochs = 200；
- warmup = 5；
- cosine decay。

所有 architecture 使用同训练 schedule。

---

# 28. Seeds

探索：

\[
0,1,2.
\]

正式：

\[
0,1,2,3,4.
\]

报告：

\[
mean\pm std.
\]

---

# 29. 主实验矩阵

第一轮建议按以下顺序：

1. Single FeedForward
2. Forward-Only Recurrent Refinement
3. Co-current + self-generated boundary
4. Countercurrent + self-generated boundary
5. Countercurrent reverse-off test
6. Null Counterflow
7. Learned-Z Counterflow

先 seed=0。

有正信号再 seeds=1,2。

---

# 30. 必须记录的主指标

- train loss；
- test/val accuracy；
- params；
- FLOPs；
- inference latency；
- peak memory；
- per-iteration accuracy；
- per-iteration confidence；
- per-iteration predictive entropy；
- q norm；
- discrepancy norm；
- H/C norm。

---

# 31. Counterflow-specific diagnostics

## 31.1 Exchange magnitude

\[
q_l^t
=
\|Q_l^t\|_2/N.
\]

看是否：

- 变成 0；
- 爆炸；
- 只某一层有 exchange；
- 各深度都存在。

---

## 31.2 Discrepancy

\[
d_l^t
=
\|D_l^t\|_2/N.
\]

画：

```text
depth -> discrepancy
```

对比：

- Co-current；
- Countercurrent。

目标不是强制 CC 一定完全 constant，而是验证它是否在深层保留更多 distributed correction signal。

---

## 31.3 Iteration accuracy

记录：

\[
Acc^0,Acc^1,\dots,Acc^T.
\]

主模型应尽量表现为：

\[
Acc^{t+1}\ge Acc^t
\]

总体趋势。

---

## 31.4 Predictive entropy

\[
\mathcal H(p^t)
=
-\sum_kp_k^t\log p_k^t.
\]

观察：

- 正确样本是否 entropy 逐步降低；
- 错误样本是否存在错误 confidence amplification。

---

# 32. Confirmation-bias diagnostics

定义：

## Error Correction Rate

对：

\[
\hat y^0\neq y
\]

的样本，统计最终：

\[
\hat y^T=y.
\]

即：

\[
ECR
=
P(\hat y^T=y\mid \hat y^0\neq y).
\]

---

## Error Amplification Rate

对初始错误样本，统计错误类别 confidence 是否继续显著增加。

例如：

\[
EAR
=
P(
p^T_{\hat y^0}
>
p^0_{\hat y^0}
+\delta
\mid
\hat y^0\neq y
).
\]

推荐：

\[
\delta=0.1.
\]

比较：

- Forward recurrent；
- feedback fusion；
- Countercurrent。

---

# 33. Reverse-Off Test

模型完整训练后：

主 inference：

\[
Q_l\neq0.
\]

然后测试：

\[
Q_l=0
\]

保持其他参数不变。

定义：

\[
\Delta_{\rm reverse}
=
Acc_{\rm full}
-
Acc_{\rm off}.
\]

如果：

\[
\Delta_{\rm reverse}\approx0
\]

说明模型没有真正依赖 reverse computation，需要谨慎。

---

# 34. 参数/FLOPs公平性

最重要：

\[
Params_{\rm CC}
=
Params_{\rm Co}.
\]

以及：

\[
FLOPs_{\rm CC}
\approx
FLOPs_{\rm Co}.
\]

另外必须提供：

- equal-compute forward recurrent baseline；
- parameter-matched single-stream baseline。

否则不能排除“只是更多计算”。

---

# 35. 推荐目录

```text
countercurrent_nn/
|
|-- models/
|   |-- stem.py
|   |-- transport.py
|   |-- boundary.py
|   |-- exchange.py
|   |-- single_stream.py
|   |-- recurrent_forward.py
|   |-- cocurrent.py
|   `-- countercurrent.py
|
|-- configs/
|   |-- cifar10_single.yaml
|   |-- cifar10_forward_recurrent.yaml
|   |-- cifar10_cocurrent_selfboundary.yaml
|   |-- cifar10_countercurrent_selfboundary.yaml
|   |-- cifar10_countercurrent_null.yaml
|   `-- cifar10_countercurrent_learnedz.yaml
|
|-- train.py
|-- evaluate.py
|-- profile_model.py
|
|-- analysis/
|   |-- plot_iteration_accuracy.py
|   |-- plot_entropy.py
|   |-- plot_exchange.py
|   |-- plot_discrepancy.py
|   |-- error_correction_analysis.py
|   `-- reverse_off_eval.py
|
`-- results/
```

---

# 36. 单元测试

## Shape

```python
assert H[l].shape == [B,64,8,8]
assert C[l].shape == [B,64,8,8]
```

---

## Conductance range

```python
gamma = 0.49 * sigmoid(logit_gamma)
assert gamma.min() > 0
assert gamma.max() < 0.5
```

---

## Paired exchange algebra

```python
h_out = h_bar - q
c_out = c_bar + q

assert_close(
    h_out + c_out,
    h_bar + c_bar
)
```

只测试 exchange substep。

---

## Source clamp

所有 refinement iteration：

```python
H0_t = stem(x)
```

必须成立。

---

## No GT leakage

主模型 `forward(x)` 不接受 label 参数。

训练 label 只能进入 loss。

---

## Reverse dependency

数值 perturbation：

改变 \(C_l\) 后应观察：

\[
H_{l+1}
\]

发生变化。

改变 \(H_l\) 后应观察：

\[
C_l
\]

发生变化。

---

# 37. Codex 最容易犯的错误

## 错误 1：只在 t=0 输入 x

之后：

```text
CL -> reverse -> forward -> output
```

脱离原输入。

禁止。

每轮必须重新：

\[
H_0=E(x).
\]

---

## 错误 2：GT 作为主 reverse boundary

禁止主模型：

```python
C_L = label_embedding(y)
```

---

## 错误 3：硬 argmax

主版本禁止：

```python
cls = logits.argmax(...)
C_L = prototype[cls]
```

使用 soft weighted hypothesis。

---

## 错误 4：reverse 只作为 decoder

如果 reverse state 没有直接改变 intermediate forward states，就不是主模型。

---

## 错误 5：最终 concat H/C

主输出只从：

\[
H_L
\]

得到。

---

## 错误 6：arbitrary attention 取代 paired flux

第一版不要 CrossAttention。

---

## 错误 7：Countercurrent 和 Co-current boundary 不同

两者必须使用同一个 self-generated target hypothesis，只改变 transport direction。

---

## 错误 8：把 predictive entropy 当成 hidden-state Shannon entropy

术语保持严谨。

---

# 38. 第一阶段 Go / No-Go

继续推进必须尽量同时看到：

### A. 性能

\[
CC>Co
\]

在多个 seed 稳定。

CIFAR-10 至少希望：

\[
+0.3\%\sim0.5\%
\]

左右稳定差异。

CIFAR-100 后希望：

\[
+0.5\%\sim1\%
\]

级别更有意义。

---

### B. Mechanism

Countercurrent 深层仍有：

\[
Q_l\neq0
\]

并保留更明显 distributed discrepancy。

---

### C. Inference necessity

Reverse-off：

\[
Acc\downarrow.
\]

---

### D. Iterative refinement

存在真实：

\[
\text{wrong at }t=0
\rightarrow
\text{correct at }t=T
\]

样本，而不只是 confidence 无脑增加。

---

# 39. 如果第一阶段 positive，下一步

1. CIFAR-100；
2. shared/mirrored transport；
3. learned PSD conductance：
   \[
   K=A^\top\Gamma A;
   \]
4. stage-wise ResNet/ConvNeXt counterflow block；
5. ImageNet-1K；
6. robustness / corruption；
7. detection / segmentation；
8. local exchange energy；
9. flux-driven local plasticity；
10. DEQ / equilibrium extension。

---

# 40. 最终实现原则

第一版主模型必须是：

\[
\boxed{
\text{fixed source evidence}
+
\text{self-generated target hypothesis}
+
\text{opposite latent transport}
+
\text{layerwise discrepancy-driven paired exchange}
+
\text{iterative reciprocal refinement}
}
\]

不要实现成：

```text
CNN -> prediction -> decoder -> concat -> classifier
```

也不要实现成：

```text
GT -> reverse network
```

主模型应体现：

> **每轮都重新看同一个输入 \(x\)，但带着上一轮形成的 target hypothesis；hypothesis 改变 evidence interpretation，而 evidence 同时修正 hypothesis。**
