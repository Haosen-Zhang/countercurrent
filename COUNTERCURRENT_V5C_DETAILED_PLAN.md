# Countercurrent V5-C：Persistent Canonical-Exchange Counterflow 详细修改方案

> **目的**
>
> 本文件用于直接指导下一版 Countercurrent 网络重构。
> 当前版本不应继续通过调 `gamma`、增加 iteration 或直接扩展到 CIFAR-100 来“碰运气”。
> 下一版首先要解决现有实现中的结构性问题，使模型真正成为：
>
> \[
> oxed{
> 	ext{persistent opposing fields}
> +	ext{fixed two-sided boundaries}
> +	ext{canonical-space paired exchange}
> +	ext{inner BVP solve}
> +	ext{outer hypothesis update}
> }
> \]
>
> 建议版本名：
>
> \[
> oxed{	extbf{V5-C: Persistent Canonical-Exchange Counterflow}}
> \]
>
> 或论文中更抽象地称为：
>
> \[
> oxed{	extbf{Canonical Counterflow Inference}}
> \]

---

# 1. 当前版本需要解决的核心问题

当前 BVP / V4 系列已经证明 exchange 参与预测，但仍存在以下结构性问题。

## 1.1 C 状态没有真正跨 iteration 持续

当前 BVP 中每个 solver sweep 都重新：

```python
boundary = make_boundary(...)
c_states = transport_c(boundary)
```

这意味着上一轮交换后得到的：

\[
C_l^{s+1}
\]

在下一轮并没有真正作为 reverse field 状态继续传播，而是被重新由 boundary 生成的整条 C 覆盖。

这不符合真正逆流换热中：

\[
oxed{
C_l^{s+1}ightarrow G_{l-1}ightarrow C_{l-1}^{s+2}
}
\]

的状态传播逻辑。

---

## 1.2 当前实现仍未真正完成“双向持续传播”

真正的 Countercurrent 不只是：

- forward 从左到右；
- reverse 从右到左；
- 每层做一次 exchange。

而应满足：

\[
oxed{
	ext{exchange 改变后的状态继续沿各自流向传播}
}
\]

即：

\[
rac{\partial H_{l+1}}{\partial C_{l+1}}
eq0
\]

并且：

\[
rac{\partial C_l}{\partial H_l}
eq0
\]

而且这两个影响不能只局限在一个 local cell，应继续进入后续 transport。

---

## 1.3 当前 “BVP boundary” 在 inner solve 中不是固定的

当前 solver 每一步都会重新根据：

\[
H_L^s
\]

计算：

\[
C_L^s=B(H_L^s)
\]

因此 solver 同时在：

1. 更新内部状态；
2. 改右侧 boundary；
3. 再更新内部状态。

这不是标准意义上的 fixed-boundary two-point BVP。

下一版必须把：

\[
oxed{	ext{Outer hypothesis update}}
\]

和：

\[
oxed{	ext{Inner fixed-boundary solve}}
\]

拆开。

---

## 1.4 Solver 当前并未稳定收敛

当前最多 8 步的 BVP refinement 中，大量 batch 并未达到：

\[
r<10^{-3}
\]

因此当前得到的更准确描述是：

\[
oxed{	ext{truncated recurrent refinement}}
\]

而不是：

\[
oxed{	ext{converged BVP solution}}
\]

下一版必须让 solver 定义与收敛判据更干净。

---

## 1.5 训练期间不应该 adaptive early-stop

当前训练和测试共用 solver，训练时也可能：

```python
if residual < tol:
    break
```

这导致训练 graph depth 随 batch 和参数动态改变。

下一版必须拆成：

```python
unroll_train(...)
solve_eval(...)
```

训练固定步数，推理才允许 adaptive stopping。

---

## 1.6 当前 target boundary 中 target semantics 太弱

已有结果显示：

- real hypothesis
- uniform hypothesis
- shuffled hypothesis

之间差异很小。

这说明：

\[
oxed{
	ext{sample-specific class hypothesis 并未成为主要 correction source}
}
\]

当前 boundary 很大程度上更像一个 shared reverse reservoir。

---

## 1.7 Class component 空间表达能力太弱

当前 class hypothesis 主要是：

\[
pZ
ightarrow Linear
ightarrow [B,C,1,1]
ightarrow broadcast\ 8	imes8
\]

因此 spatial dimension 上是常量。

这对 CNN feature map 来说表达能力不足。

---

## 1.8 H 和 C 可能根本不在同一个坐标系

当前 P1 中：

\[
F_l
\]

和：

\[
G_l
\]

是独立学习的。

因此 H 和 C 可能编码在不同 latent basis 中。

即使语义相同，也可能存在：

\[
C=RH
\]

其中 \(R\) 是旋转或一般线性变换。

此时直接：

\[
D=H-C
\]

并没有明确语义。

---

## 1.9 现有 exchange 可能退化成简单缩放

早期诊断出现过：

\[
\cos(D,H)pprox0.99
\]

说明：

\[
Dpprox H
\]

从而：

\[
Q=\gamma D
\]

实际上近似：

\[
H'pprox(1-\gamma)H
\]

这不是“纠正方向”，而只是激活缩放。

---

## 1.10 参数匹配不等于 forward capacity 匹配

普通 CNN 把全部参数用于：

\[
xightarrow y
\]

主路径。

Counterflow 则把大量参数分给 reverse branch。

因此即使参数总量一致，forward representation 本身可能更弱。

这也是为什么 t=0 结果常常低于参数匹配 CNN。

---

# 2. V5-C 总体设计目标

V5-C 不再把“逆流”定义为：

> 一个 forward CNN + 一个 reverse CNN + 局部 feature mixing。

而定义为：

\[
oxed{
	ext{两个由相反边界驱动的 persistent fields}
}
\]

它们：

1. 从相反边界进入；
2. 沿相反方向 transport；
3. 在共同 canonical exchange space 中发生 paired exchange；
4. exchanged states 持续进入下一次 inner iteration；
5. inner solve 期间两端 boundary 固定；
6. outer loop 才更新 target hypothesis。

---

# 3. 两层时间尺度：Outer / Inner

整个 inference 拆成：

\[
oxed{
	ext{Outer hypothesis loop}
+
	ext{Inner fixed-boundary counterflow solver}
}
\]

---

# 4. Outer Loop：负责更新 target hypothesis

定义 outer iteration：

\[
k=0,1,\dots,K_{m outer}-1
\]

当前 terminal forward state：

\[
H_L^{(k)}
\]

得到 logits：

\[
s^{(k)}=Head(H_L^{(k)})
\]

以及：

\[
p^{(k)}
=
\operatorname{softmax}
\left(
rac{s^{(k)}}{	au}
ight)
\]

然后生成 target-side boundary：

\[
oxed{
C_L^{(k)}
=
B(p^{(k)},H_L^{(k)})
}
\]

**进入 inner solver 后，\(C_L^{(k)}\) 必须固定。**

---

# 5. Inner Loop：真正求 fixed-boundary Countercurrent BVP

在 outer step \(k\) 内：

source boundary：

\[
oxed{
H_0=E(x)
}
\]

target boundary：

\[
oxed{
C_L=C_L^{(k)}
}
\]

两者在整个 inner solver 中固定。

待求状态：

\[
H_1,\dots,H_L
\]

和：

\[
C_0,\dots,C_{L-1}
\]

---

# 6. C 状态必须真正 persistent

Inner iteration：

\[
s=0,1,\dots,S-1
\]

必须保留：

\[
H^s,C^s
\]

并计算：

\[
H^{s+1},C^{s+1}
\]

下一步直接读取上一轮的 C：

\[
oxed{
ar C_l^s=G_l(C_{l+1}^s)
}
\]

而不是重新：

\[
C^s\leftarrow Transport(B)
\]

---

# 7. 第一版推荐使用 Jacobi-style synchronous solver

为了避免 layer update order bias，第一版主实现建议使用同步 Jacobi。

即第 \(s+1\) 步的所有 cell：

\[
H^{s+1},C^{s+1}
\]

都只读取：

\[
H^s,C^s
\]

。

---

# 8. 每层 transport

对于：

\[
l=0,\dots,L-1
\]

forward transport：

\[
oxed{
ar H_{l+1}^s
=
F_l(H_l^s)
}
\]

countercurrent reverse transport：

\[
oxed{
ar C_l^s
=
G_l(C_{l+1}^s)
}
\]

这里 reverse stream 是：

\[
Lightarrow0
\]

。

---

# 9. V5-C 核心：Canonical Exchange Space

原始 C 方案需要升级成：

> H 和 C 可以保留各自的 representation basis，但 exchange 前分别映射到同一个 canonical exchange space。

定义：

\[
A_l^H
\]

和：

\[
A_l^C
\]

将 H/C 映射到：

\[
\mathcal U_l
\]

：

\[
oxed{
u_H=A_l^Har H_{l+1}
}
\]

\[
oxed{
u_C=A_l^Car C_l
}
\]

。

---

# 10. 为什么不能简单用同一个 \(\phi(H),\phi(C)\)

如果：

\[
C=RH
\]

而 R 是非平凡 basis rotation，

那么使用同一个：

\[
\phi(H),\phi(C)
\]

并不能保证对齐。

所以需要：

\[
A_H
\]

和：

\[
A_C
\]

分别把两种 representation 投入一个共同 canonical space。

---

# 11. Canonical transforms 的约束

建议：

\[
A_l^H,A_l^C
\]

使用正交变换：

\[
(A_l^H)^	op A_l^H=I
\]

\[
(A_l^C)^	op A_l^C=I
\]

。

这样它们只重新选择坐标基，不随意改变 feature norm。

初始化：

\[
A_l^H=I
\]

\[
A_l^C=I
\]

。

---

# 12. Canonical discrepancy

真正的 discrepancy 定义为：

\[
oxed{
D_l
=
u_H-u_C
}
\]

而不再直接使用：

\[
H-C.
\]

此时：

\[
D_l
\]

可解释为：

\[
oxed{
	ext{cross-stream inconsistency in a shared semantic coordinate system}
}
\]

。

---

# 13. Paired exchange 在 canonical space 中进行

定义 conductance：

\[
\Gamma_l
=
\operatorname{diag}(\gamma_l)
\]

其中：

\[
0<\gamma_{l,c}<0.5
\]

。

exchange flux：

\[
oxed{
Q_l
=
\Gamma_lD_l
}
\]

。

canonical states 更新：

\[
oxed{
u_H^+=u_H-Q_l
}
\]

\[
oxed{
u_C^+=u_C+Q_l
}
\]

因此：

\[
u_H^++u_C^+=u_H+u_C
\]

是 canonical exchange substep 内的 paired antisymmetric update。

---

# 14. Decode 回各自 representation

因为 \(A_H,A_C\) 正交：

\[
A^{-1}=A^	op
\]

所以：

\[
oxed{
\hat H_{l+1}
=
(A_l^H)^	op u_H^+
}
\]

\[
oxed{
\hat C_l
=
(A_l^C)^	op u_C^+
}
\]

。

---

# 15. 完整 Countercurrent Cell

\[
oxed{
egin{aligned}
ar H_{l+1}&=F_l(H_l)\
ar C_l&=G_l(C_{l+1})\
u_H&=A_l^Har H_{l+1}\
u_C&=A_l^Car C_l\
D_l&=u_H-u_C\
Q_l&=\Gamma_lD_l\
u_H^+&=u_H-Q_l\
u_C^+&=u_C+Q_l\
\hat H_{l+1}&=(A_l^H)^	op u_H^+\
\hat C_l&=(A_l^C)^	op u_C^+
\end{aligned}
}
\]

---

# 16. Solver damping 与 gamma 必须分开

gamma 是物理类比中的 exchange conductance。

solver damping 是数值求解参数。

不能再用 gamma 来兼任 solver stabilization。

定义：

\[
ho\in(0,1]
\]

。

更新：

\[
oxed{
H_{l+1}^{s+1}
=
(1-ho)H_{l+1}^s
+
ho\hat H_{l+1}
}
\]

\[
oxed{
C_l^{s+1}
=
(1-ho)C_l^s
+
ho\hat C_l
}
\]

第一版建议：

\[
oxed{ho=0.5}
\]

---

# 17. 每个 inner step 重新 clamp 两个 boundary

每次更新结束后：

\[
oxed{
H_0^{s+1}=E(x)
}
\]

\[
oxed{
C_L^{s+1}=C_L^{(k)}
}
\]

。

注意：

`stem(x)` 可以只算一次：

```python
source = stem(x)
```

然后每个 inner step：

```python
H_new[0] = source
```

即可。

---

# 18. Target Boundary 重新设计为三个分量

主 boundary：

\[
oxed{
C_L
=
g_B
\left(
\lambda_0 C_{m base}
+
\lambda_c C_{m class}
+
\lambda_i C_{m instance}
ight)
}
\]

三个 component 必须独立记录、独立消融。

---

# 19. Base component

定义：

\[
C_{m base}
\in
\mathbb R^{1	imes C	imes h	imes w}
\]

。

作用：

> 表示 task-level 的共同 target-side reservoir。

建议初始：

\[
\lambda_0=0.1\sim0.2
\]

。

---

# 20. Class component：必须中心化 probability

先：

\[
p=\operatorname{softmax}(s)
\]

定义：

\[
oxed{
	ilde p
=
p-rac1K\mathbf 1
}
\]

。

这样 uniform hypothesis：

\[
p_k=rac1K
\]

时：

\[
oxed{
C_{m class}=0
}
\]

。

这可以直接消除 prototype 的 common-mode component。

---

# 21. Class component：必须 spatial

推荐低秩 class embedding：

\[
E_{m cls}
\in
\mathbb R^{K	imes r}
\]

例如：

\[
r=16
\]

。

然后：

\[
z_{m cls}
=
	ilde pE_{m cls}
\]

。

通过 spatial projector：

\[
W_{m spatial}:
\mathbb R^r
ightarrow
\mathbb R^{C	imes h	imes w}
\]

得到：

\[
oxed{
C_{m class}
=
reshape(W_{m spatial}z_{m cls})
}
\]

。

不要再使用简单 `[B,C,1,1]` broadcast 作为主版本。

---

# 22. Centered class hypothesis 还自动提供 uncertainty gating

如果 prediction 很不确定：

\[
ppproxrac1K
\]

则：

\[
	ilde ppprox0
\]

所以：

\[
C_{m class}pprox0
\]

。

因此：

> 模型自己都不确定时，不会产生强 class target forcing。

第一版不必另加 confidence gate。

---

# 23. Instance component

定义：

\[
oxed{
C_{m instance}
=
I(\operatorname{sg}(H_L))
}
\]

其中：

\[
\operatorname{sg}
\]

表示 stop-gradient。

推荐：

```text
1×1 Conv
GN
SiLU
3×3 Conv
```

输出：

\[
[B,C,h,w]
\]

。

---

# 24. 防止 instance component 吞掉 class semantics

建议初始化：

\[
\lambda_c=1
\]

\[
\lambda_i=0.1
\]

\[
\lambda_0=0.1\sim0.2
\]

。

每个 checkpoint 都记录：

\[
\|\lambda_0C_{m base}\|
\]

\[
\|\lambda_cC_{m class}\|
\]

\[
\|\lambda_iC_{m instance}\|
\]

。

如果：

\[
C_{m instance}\gg C_{m class}
\]

说明系统又退化成普通 feature echo。

---

# 25. Global boundary gain

保留：

\[
g_B>0
\]

用于初始化 scale calibration：

\[
RMS(C_L)pprox RMS(H_L)
\]

。

它只负责尺度，不负责 component semantics。

---

# 26. Outer / Inner 完整流程

## Initialization

计算：

\[
S=E(x)
\]

pure forward：

\[
H_0=S
\]

\[
H_{l+1}=F_l(H_l)
\]

得到：

\[
H_L^0
\]

以及：

\[
p^0=\operatorname{softmax}(Head(H_L^0))
\]

。

---

# 27. Outer step \(k\)

构造：

\[
oxed{
C_L^{(k)}
=
B(p^k,\operatorname{sg}(H_L^k))
}
\]

。

然后固定这个 boundary。

---

# 28. Inner state initialization

H：

- 第一次 outer 使用 pure-forward states；
- 后续 outer 可使用上一 outer equilibrium 作为 warm start。

C：

第一次可以从 boundary 做一次 reverse transport 仅作为初始化：

\[
C_l^0=G_l(C_{l+1}^0)
\]

。

注意：

> 这只是 initialization。

进入 inner loop 后，C 必须 persistent。

---

# 29. Inner solve

对：

\[
s=0,\dots,S_{m inner}-1
\]

执行：

\[
ar H_{l+1}=F_l(H_l^s)
\]

\[
ar C_l=G_l(C_{l+1}^s)
\]

然后 canonical paired exchange，得到：

\[
\hat H,\hat C
\]

。

再 damping：

\[
H^{s+1}
=
(1-ho)H^s+ho\hat H
\]

\[
C^{s+1}
=
(1-ho)C^s+ho\hat C
\]

。

最后 clamp：

\[
H_0^{s+1}=S
\]

\[
C_L^{s+1}=C_L^{(k)}
\]

。

---

# 30. Equation residual

不要只使用 step change。

定义 fixed-point operator：

\[
T(H,C)
=
(\hat H,\hat C)
\]

。

使用 undamped equation residual：

\[
oxed{
r_{m eq}
=
\sqrt{
rac{
\|T_H(H,C)-H\|_2^2
+
\|T_C(H,C)-C\|_2^2
}{
\|H\|_2^2+\|C\|_2^2+\epsilon
}
}
}
\]

只包含 interior states：

\[
H_1,\dots,H_L
\]

和：

\[
C_0,\dots,C_{L-1}
\]

。

不要包含固定 boundary：

\[
H_0,C_L.
\]

---

# 31. 为什么不能只看 step residual

因为：

\[
S^{s+1}
=
S^s+ho(T(S^s)-S^s)
\]

所以：

\[
\|S^{s+1}-S^s\|
=
ho\|T(S^s)-S^s\|
\]

。

当 \(ho\) 很小时，step change 会人为显得很小。

因此 stopping criterion 应使用：

\[
r_{m eq}
\]

。

---

# 32. Training 与 Evaluation 必须分离

## Training

实现：

```python
unroll_train(...)
```

固定：

\[
S_{m train}=8
\]

。

绝对不允许 adaptive early stop。

即使：

\[
r_{m eq}<10^{-3}
\]

也继续完成固定 8 步。

---

## Evaluation

实现：

```python
solve_eval(...)
```

才允许：

\[
r_{m eq}<\epsilon
\]

提前结束。

建议：

\[
S_{\max}=24
\]

\[
\epsilon=10^{-3}
\]

。

---

# 33. Eval 使用 per-sample stopping

不要再按 batch 判断是否收敛。

对每个 sample \(b\)：

\[
r_b
\]

单独计算。

当：

\[
r_b<\epsilon
\]

该 sample freeze。

其他 sample 继续。

报告：

- mean steps
- median steps
- P90
- P95
- 未收敛比例

---

# 34. 第一版删除 truncate_steps

当前 architecture repair 阶段不要同时引入 truncated BPTT。

建议：

\[
oxed{	ext{full unroll 8 steps}}
\]

。

如果显存不够，优先用：

```python
torch.utils.checkpoint
```

而不是截断梯度。

后续再研究：

- TBPTT
- phantom gradient
- implicit differentiation
- DEQ adjoint

---

# 35. Outer loop 第一阶段只做 1 次

第一轮建议：

\[
oxed{
K_{m outer}=1
}
\]

即：

```text
pure forward
   ↓
target hypothesis
   ↓
fixed target boundary
   ↓
true counterflow BVP solve
   ↓
final output
```

。

先证明 fixed-boundary counterflow 本身有效。

---

# 36. 第二阶段才测试 outer=2

若 outer=1 有正信号，再：

\[
p^0
ightarrow B^0
ightarrow BVP
ightarrow p^1
ightarrow B^1
ightarrow BVP
ightarrow p^2
\]

。

此时才研究 hypothesis revision 的额外价值。

---

# 37. Co-current Control 必须严格镜像

Co-current 使用完全相同：

- boundary decomposition；
- canonical transforms；
- gamma；
- damping；
- solver；
- train steps；
- eval tolerance；
- parameter count。

唯一改变：

Countercurrent：

\[
C:Lightarrow0
\]

Co-current：

\[
C:0ightarrow L
\]

。

---

# 38. Countercurrent Cell pairing

Countercurrent：

\[
ar H_{l+1}=F_l(H_l)
\]

\[
ar C_l=G_l(C_{l+1})
\]

然后 exchange。

---

# 39. Co-current Cell pairing

Co-current：

\[
ar H_{l+1}=F_l(H_l)
\]

\[
ar C_{l+1}=G_l(C_l)
\]

然后在对应共同 depth 上使用完全相同 canonical exchange operator。

---

# 40. Gamma 第一版保持简单

继续：

\[
\gamma_l
=
0.49\sigma(a_l)
\]

初始化：

\[
a_l=-2.2
\]

即：

\[
\gammapprox0.049
\]

。

必须：

\[
oxed{	ext{gamma weight decay}=0}
\]

。

第一版不要 adaptive gamma。

---

# 41. Adaptive Conductivity 放到后续

如果 V5-C 本身成立，之后才考虑：

\[
\gamma_l(x)
=
0.49\sigma(
a_l+f_l(|D_l|)
)
\]

或者 confidence-aware conductance。

不要和 architecture repair 混在一起。

---

# 42. Training Loss 第一版只用最终 CE

使用：

\[
oxed{
L=CE(\hat y,y)
}
\]

。

不要：

- initial CE
- local energy loss
- discrepancy loss
- solver residual loss

第一版只把这些量作为 diagnostics。

---

# 43. Solver 不稳定时优先调 damping

建议默认：

\[
ho=0.5
\]

。

如果不稳定，再比较：

\[
0.25,\ 0.5,\ 0.75,\ 1.0
\]

。

不要为了收敛重新调 gamma。

---

# 44. 必须记录的 V5-C diagnostics

每个 inner step：

\[
r_{m eq}
\]

\[
r_{m step}
\]

。

每个 layer：

\[
\|H_l\|
\]

\[
\|C_l\|
\]

\[
\|u_H\|
\]

\[
\|u_C\|
\]

\[
\|D_l\|
\]

\[
\|Q_l\|
\]

。

---

# 45. Canonical-space 方向诊断

记录：

\[
\cos(u_H,u_C)
\]

\[
\cos(D_l,u_H)
\]

\[
\cos(Q_l,u_H)
\]

。

关键判断：

如果：

\[
\cos(D_l,u_H)pprox1
\]

仍长期成立，

说明 exchange 又退化成 H scaling。

---

# 46. Boundary component diagnostics

定义：

\[
R_{m base}
=
rac{\|\lambda_0C_{m base}\|}{\|C_L\|}
\]

\[
R_{m class}
=
rac{\|\lambda_cC_{m class}\|}{\|C_L\|}
\]

\[
R_{m inst}
=
rac{\|\lambda_iC_{m instance}\|}{\|C_L\|}
\]

。

如果：

\[
R_{m class}pprox0
\]

说明 target hypothesis 仍未被使用。

---

# 47. Boundary intervention 必须重新做

冻结 checkpoint 比较：

1. real hypothesis
2. uniform hypothesis
3. shuffled hypothesis
4. class-off
5. instance-off
6. base-only
7. zero boundary

由于 centered probability：

\[
p=rac1K
\Rightarrow
C_{m class}=0
\]

因此解释会比旧版本干净得多。

---

# 48. 新增：C Persistence Ablation

主模型：

\[
C^{s+1}
\]

持续进入下一步。

Ablation：

每个 step 强行：

\[
C\leftarrow Transport(B)
\]

。

比较：

\[
oxed{
Persistent\ C
\quad vs\quad
Reset\ C
}
\]

。

如果前者更好，直接说明真正 transported reverse state 有价值。

---

# 49. 新增：Fixed vs Moving Boundary Ablation

主模型：

Inner solve 中：

\[
C_L=C_L^{(k)}
\]

固定。

Ablation：

每个 inner step 重新：

\[
C_L^s=B(p^s,H_L^s)
\]

。

比较：

\[
oxed{
Fixed-boundary BVP
}
\]

vs

\[
oxed{
Moving-boundary recurrent feedback
}
\]

。

---

# 50. 新增：Raw vs Canonical Exchange Ablation

A：

\[
D=H-C
\]

。

B：

\[
D=A_HH-A_CC
\]

。

保持其余完全相同。

比较：

\[
oxed{
CanonicalExchange
-
RawExchange
}
\]

可直接验证 representation alignment 是否必要。

---

# 51. 文件级修改建议

不要直接覆盖旧 BVP，以保留复现实验能力。

新增：

```text
countercurrent_nn/models/
    canonical_exchange.py
    decomposed_boundary.py
    persistent_bvp.py
```

。

旧：

```text
bvp_lattice.py
```

保留为 legacy / P1 reproduction。

---

# 52. `canonical_exchange.py`

实现：

```python
class CanonicalExchange(nn.Module):
    ...
```

建议接口：

```python
h_new, c_new, diagnostics = exchange(
    h_bar,
    c_bar,
)
```

内部：

```python
u_h = encode_h(h_bar)
u_c = encode_c(c_bar)

d = u_h - u_c
q = gamma * d

u_h_new = u_h - q
u_c_new = u_c + q

h_new = decode_h(u_h_new)
c_new = decode_c(u_c_new)
```

---

# 53. Orthogonal transform 实现建议

可使用：

```python
nn.Linear(C, C, bias=False)
```

并加：

```python
torch.nn.utils.parametrizations.orthogonal(...)
```

初始化为 identity。

feature map：

```text
[B,C,H,W]
 -> [B,H,W,C]
 -> Linear
 -> [B,C,H,W]
```

或使用 `einsum`。

---

# 54. `decomposed_boundary.py`

实现：

```python
class DecomposedTargetBoundary(nn.Module):
    ...
```

包含：

```text
base_state
class_embedding
class_spatial_projector
instance_encoder
base_scale
class_scale
instance_scale
global_gain
```

核心 forward：

```python
p = softmax(logits)
p_centered = p - 1.0 / num_classes

z_cls = p_centered @ class_embedding
c_cls = spatial_projector(z_cls)
c_cls = c_cls.view(B, C, H, W)

c_inst = instance_encoder(h_terminal.detach())

boundary = (
    lambda_base * c_base
    + lambda_class * c_cls
    + lambda_instance * c_inst
)

boundary = global_gain * boundary
```

---

# 55. `persistent_bvp.py`

新增：

```python
class PersistentCanonicalCounterflow(nn.Module):
    ...
```

以及：

```python
class PersistentCanonicalCocurrent(nn.Module):
    ...
```

。

必须提供：

```python
unroll_train(...)
solve_eval(...)
```

两个独立入口。

---

# 56. Training pseudo-code

```python
source = stem(x)

# initial forward
H = pure_forward_states(source)
logits = classify(H[-1])

C = None

for outer in range(num_outer_train):

    boundary = target_boundary(
        logits,
        H[-1],
    )

    if C is None:
        C = initialize_reverse_states(boundary)

    for inner in range(train_inner_steps):

        H_candidate, C_candidate = fixed_point_map(
            H,
            C,
            source=source,
            boundary=boundary,
        )

        H = damp(H, H_candidate, rho)
        C = damp(C, C_candidate, rho)

        H[0] = source
        C[L] = boundary

    logits = classify(H[-1])

return logits
```

---

# 57. Eval pseudo-code

```python
source = stem(x)

H = pure_forward_states(source)
logits = classify(H[-1])

for outer in range(num_outer_eval):

    boundary = target_boundary(logits, H[-1])

    C = initialize_or_warmstart_C(C, boundary)

    active = torch.ones(B, dtype=bool)

    for inner in range(eval_max_steps):

        H_candidate, C_candidate = fixed_point_map(
            H,
            C,
            source=source,
            boundary=boundary,
        )

        residual = equation_residual_per_sample(
            H,
            C,
            H_candidate,
            C_candidate,
        )

        update only active samples

        freeze samples where residual < tol

        H[0] = source
        C[L] = boundary

        if all converged:
            break

    logits = classify(H[-1])
```

---

# 58. 推荐第一版配置

| 参数 | 建议值 |
|---|---:|
| Dataset | CIFAR-10 |
| channels | 64 |
| depth | 4 |
| outer train | 1 |
| outer eval | 1 |
| inner train steps | 8 |
| eval max steps | 24 |
| tolerance | \(10^{-3}\) |
| damping | 0.5 |
| gamma init | ≈0.05 |
| gamma max | 0.49 |
| class rank | 16 |
| class scale | 1.0 |
| instance scale | 0.1 |
| base scale | 0.1–0.2 |
| loss | final CE |
| gamma WD | 0 |
| adaptive train stop | off |
| adaptive eval stop | on |
| F/G | independent initially |

---

# 59. 为什么第一版仍保留独立 F/G

当前先只验证：

1. persistent solver；
2. fixed boundary；
3. canonical exchange；
4. decomposed target boundary。

如果同时把：

\[
Gpprox F^{-1}
\]

或 F/G sharing 也加入，最终 improvement 很难归因。

因此 V5-C 第一版：

\[
oxed{
F/G	ext{ 保持独立}
}
\]

。

---

# 60. 后续才处理 forward capacity / 参数效率

如果 V5-C 能稳定：

\[
CC>Co
\]

但：

\[
CC<CNN
\]

则说明 counterflow inductive bias 有效，但参数使用效率不好。

第二阶段再做：

\[
F/G	ext{ shared weights}
\]

或者：

\[
G=F+	ext{small reverse adapter}
\]

或：

\[
Gpprox F^{-1}
\]

。

---

# 61. Unit Tests 必须新增的行为测试

## 61.1 C persistence test

人为扰动：

\[
C_l^s
\]

必须影响：

\[
C_{l-1}^{s+1}
\]

且下一步不能被 boundary-generated C 覆盖。

---

## 61.2 Boundary call-count test

若：

\[
K_{m outer}=1
\]

inner 8 steps，

boundary module 应只调用：

\[
1
\]

次。

---

## 61.3 Source clamp test

每个 inner step：

\[
H_0=E(x)
\]

必须成立。

---

## 61.4 Target clamp test

每个 inner step：

\[
C_L=C_L^{(k)}
\]

必须完全一致。

---

## 61.5 Canonical antisymmetry

必须：

\[
u_H^++u_C^+
=
u_H+u_C
\]

数值误差范围内成立。

---

## 61.6 Orthogonality

必须：

\[
A^	op Approx I
\]

。

---

## 61.7 Uniform class test

若：

\[
p_k=1/K
\]

则：

\[
C_{m class}=0
\]

。

---

## 61.8 Spatial class test

不同 spatial location 的：

\[
C_{m class}
\]

不能全部相同。

---

## 61.9 Train fixed-step test

即使第一 inner step：

\[
r<tol
\]

training 仍执行完整：

\[
S_{m train}
\]

步。

---

## 61.10 Eval adaptive-stop test

eval 才允许提前结束。

---

## 61.11 No GT leakage

主模型：

```python
forward(x)
```

不接受 label。

---

# 62. 正式训练前增加线性 Toy-BVP Test

先构造线性系统：

\[
H_{l+1}=AH_l-Q_l
\]

\[
C_l=BC_{l+1}+Q_l
\]

。

要求：

1. solver 残差稳定下降；
2. persistent C 确实传播；
3. fixed boundary 正确；
4. 数值解逼近直接线性方程解；
5. damping 不改变最终 fixed point；
6. eval early-stop 与固定长迭代一致。

如果 toy system 都不稳定，禁止直接跑 200 epochs CIFAR-10。

---

# 63. CIFAR-10 实验顺序

## Experiment A

Persistent fixed-boundary solver + raw exchange：

\[
D=H-C
\]

。

目的：

> 单独测试 persistence / solver 修复是否有价值。

---

## Experiment B

Persistent fixed-boundary solver + Canonical Exchange：

\[
D=A_HH-A_CC
\]

。

目的：

> 验证 representation alignment。

---

## Experiment C

Co-current Canonical control。

目的：

> 测试 opposing transport geometry。

---

## Experiment D

Boundary decomposition ablations。

---

## Experiment E

Persistence ablation。

---

## Experiment F

Fixed vs moving boundary ablation。

---

# 64. 第一阶段 diagnostics 必须优先于跑更多 seed

每个实验先 seed 0。

先确认：

- solver 真收敛；
- C 真 persistent；
- class boundary 真有语义；
- D 不再退化成 H scaling；
- CC/Co dynamics 真有差异。

这些成立后再 seed 1/2。

---

# 65. 新 Go / No-Go：CIFAR-100 前的四道门

## Gate 1：Solver 收敛

希望：

\[
>95\%
\]

test samples 在 24 inner steps 内达到：

\[
r_{m eq}<10^{-3}
\]

。

---

## Gate 2：Target hypothesis 有实际语义贡献

要求真实 hypothesis 明显优于：

- uniform
- shuffled
- class-off

不能继续只有：

\[
0.08\%\sim0.1\%
\]

级别的差异。

---

## Gate 3：Countercurrent 稳定优于 Co-current

至少：

\[
3	ext{ seeds}
\]

方向一致：

\[
oxed{
CC>Co
}
\]

。

这是最重要的 topology evidence。

---

## Gate 4：Exchange 不再退化为 scaling

不能长期：

\[
\cos(D,u_H)pprox1
\]

。

需要证明：

\[
D
\]

是真实 correction direction，而不是 forward activation 本身。

---

# 66. 什么时候再要求超过 CNN

第一阶段先证明：

\[
CC>Co
\]

以及：

\[
	ext{target semantics matter}
\]

。

如果：

\[
CC>Co
\]

但：

\[
CC<CNN
\]

说明 architecture principle 可能成立，但 parameter / compute efficiency 还需要优化。

这时再处理：

- F/G sharing；
- reverse adapter；
- inverse transport；
- equal-compute redesign。

---

# 67. 真正的 falsification 条件

只有当以下条件都满足：

1. persistent H/C solver 已正确实现；
2. fixed boundary inner solve 已正确实现；
3. canonical exchange 已解决 basis mismatch；
4. target boundary semantics 已增强；
5. solver 已真正收敛；
6. 多 seed；
7. CC / Co 公平匹配；

之后仍然：

\[
oxed{
CC\le Co
}
\]

且 real / uniform / shuffled boundary 没明显区别，

才可以认真认为：

\[
oxed{
	ext{opposing-flow inductive bias 在 CIFAR classification 上可能本身无优势}
}
\]

。

---

# 68. 最终 architecture summary

V5-C 的核心不是：

```text
Forward CNN
+
Reverse CNN
+
Feature Fusion
```

而是：

```text
             OUTER LOOP

      H_L -> target hypothesis
                  |
                  v
              fixed C_L
                  |
                  v

        INNER COUNTERFLOW SOLVER

source H_0 ======================= target C_L

H0  -> H1  -> H2  -> ... -> HL
      ↕      ↕              ↕
C0  <- C1  <- C2  <- ... <- CL

Each cell:
    F/G transport
        ↓
canonical alignment
        ↓
D = u_H - u_C
        ↓
Q = Gamma D
        ↓
paired (-Q,+Q)
        ↓
persistent H/C states
```

---

# 69. Counterflow Cell summary

\[
oxed{
egin{aligned}
ar H_{l+1}&=F_l(H_l^s)\
ar C_l&=G_l(C_{l+1}^s)\
u_H&=A_l^Har H_{l+1}\
u_C&=A_l^Car C_l\
D_l&=u_H-u_C\
Q_l&=\Gamma_lD_l\
u_H^+&=u_H-Q_l\
u_C^+&=u_C+Q_l\
\hat H_{l+1}&=(A_l^H)^	op u_H^+\
\hat C_l&=(A_l^C)^	op u_C^+\
H_{l+1}^{s+1}
&=(1-ho)H_{l+1}^s+ho\hat H_{l+1}\
C_l^{s+1}
&=(1-ho)C_l^s+ho\hat C_l
\end{aligned}
}
\]

with hard boundary clamp:

\[
oxed{
H_0^{s+1}=E(x)
}
\]

\[
oxed{
C_L^{s+1}=C_L^{(k)}
}
\]

。

---

# 70. 最重要的实现原则

下一版必须满足：

1. **C 是 persistent state，不再每步重建整条 C。**
2. **Inner solve 期间 target boundary 固定。**
3. **Outer loop 才更新 target hypothesis。**
4. **Train 固定步数，Eval 才 adaptive early-stop。**
5. **Residual 使用 undamped equation residual。**
6. **H/C 在 canonical exchange space 中比较。**
7. **Boundary 拆成 base/class/instance 三部分。**
8. **Class probability 必须中心化。**
9. **Class boundary 必须具有 spatial structure。**
10. **Instance branch 必须 stop-gradient，并限制初始强度。**
11. **Paired exchange 仍保持同一个 Q 的 \(-Q,+Q\)。**
12. **Co-current 使用完全相同的 solver/boundary/exchange，只改 transport direction。**
13. **第一版不要同时改 F/G sharing、adaptive gamma、loss 或 no-BP。**
14. **先通过 toy-BVP 和 CIFAR-10 机制 gate，再考虑 CIFAR-100。**

---

# 71. 当前建议的研究路线

\[
oxed{
	ext{V5-C solver correctness}
ightarrow
	ext{Canonical exchange validity}
ightarrow
	ext{CC vs Co}
ightarrow
	ext{Boundary semantics}
ightarrow
	ext{Multi-seed}
ightarrow
	ext{Forward-capacity optimization}
ightarrow
	ext{CIFAR-100}
}
\]

不要再走：

\[
	ext{CIFAR-10 不理想}
ightarrow
	ext{继续调 gamma}
ightarrow
	ext{直接 CIFAR-100}
\]

。

这一版的目标不是马上刷高 accuracy，而是先把真正的 Countercurrent computation 定义和实现干净。
