# Countercurrent Neural System：核心思想与研究约束

> **用途**：本文件用于给没有当前对话上下文的实现/研究 Agent（例如 Codex）说明研究思想、理论边界、与已有 Counter-Current Learning (CCL) 的区别，以及后续实验必须遵守的设计原则。  
> **优先级**：如果工程实现与本文件冲突，应优先保持本文件中的科学问题，而不是为了方便把模型改成普通双流网络、encoder-decoder、双向 RNN、简单 feedback 或 feature fusion。

---

# 0. 一句话研究问题

我们希望研究：

> **神经网络能否把“逆流交换”从一种训练启发，变成 inference 本身的计算原则：source-derived forward state 与 target-hypothesis-derived reverse state 从网络深度的相反边界进入、沿相反方向传播，并在每个局部 layer 通过 discrepancy-driven paired exchange 互相修正，最终形成一个 evidence–hypothesis reconciliation system？**

核心不是“为了反向而反向”，而是：

\[
\boxed{
\text{source evidence}
+
\text{target-side hypothesis}
+
\text{opposite transport}
+
\text{local paired exchange}
+
\text{iterative reciprocal correction}
}
\]

最终想研究的是：

\[
\boxed{
\text{Can counterflow geometry define the computation itself?}
}
\]

---

# 1. 首先明确：不要再包装成“物理时间反演”

本项目不主张：

\[
t\rightarrow -t
\]

意义上的 time-reversal symmetry。

原因：

- 网络 depth 的反向传播不等价于物理时间反演；
- 真实逆流换热本身是耗散系统，并不是时间可逆系统；
- RevNet / Neural ODE / Hamiltonian NN 等领域对 “reversibility/time-reversal” 已有严格定义；
- 为了噱头声称“时空对称”会降低理论严谨性。

我们可以使用：

- **counter-propagating latent fields**
- **two-boundary inference**
- **reciprocal inference**
- **depth-space counterflow**
- **boundary-driven non-equilibrium neural system**

等更准确的术语。

如果定义 computational spacetime，则：

- depth \(l\)：计算空间坐标；
- iteration \(t\)：relaxation / inference time；

但不声称物理 time-reversal symmetry。

---

# 2. 逆流换热真正映射到神经网络的是什么

物理逆流换热的核心不是 hot/cold 标签本身，而是：

\[
\boxed{
\text{opposite transport}
+
\text{two-side inlet boundaries}
+
\text{local exchange}
+
\text{distributed driving difference}
}
\]

神经网络版本：

正向状态：

\[
H_0 \rightarrow H_1 \rightarrow \cdots \rightarrow H_L
\]

逆向状态：

\[
C_0 \leftarrow C_1 \leftarrow \cdots \leftarrow C_L
\]

其中：

\[
H_0 = E_x(x)
\]

是 source-side evidence boundary。

而：

\[
C_L = B(z)
\]

是 target-side boundary。

---

# 3. 物理方向的正确理解

以共同的 depth 轴 \(0\rightarrow L\) 表示：

```text
source side                                      target side

Forward:
H0  -------------------------------------------> HL

Reverse:
C0  <------------------------------------------- CL
```

如果用 hot/cold 只作为换热类比：

```text
hot  -------------------------------------------> cold(er)
hotter <------------------------------------------ cold
```

逆向流的入口在右端，是 cold-side inlet。

因此如果把网络右端解释成 target-side boundary，则：

\[
\boxed{C_L\text{ 应该是 target-side state}}
\]

而不是“为了和 target 相反而塞一个 opposite state”。

---

# 4. 分类任务中 Forward / Reverse 的语义

## Forward stream

\[
H_0=E_x(x)
\]

从输入 evidence 出发：

\[
x\rightarrow H_1\rightarrow\cdots\rightarrow H_L.
\]

它回答：

> **Given what I observe, what output-side representation is supported?**

---

## Reverse stream

\[
C_L=B(z)
\]

从当前 target hypothesis 出发：

\[
C_L\rightarrow C_{L-1}\rightarrow\cdots\rightarrow C_0.
\]

它回答：

> **If the current output hypothesis were approximately correct, what representations should be expected at earlier depths?**

所以本项目的核心不是：

\[
\text{forward evidence} \leftarrow \text{fixed target}
\]

而是：

\[
\boxed{
\text{forward evidence}
\leftrightarrow
\text{current target hypothesis}
}
\]

---

# 5. 关键升级：主模型的 \(C_L\) 不使用 GT，而来自当前 \(H_L\)

训练时使用 GT 作为 reverse boundary：

\[
C_L=B(y_{\rm GT})
\]

会非常接近 Kao & Hariharan, NeurIPS 2024 的 Counter-Current Learning (CCL)。

CCL 的核心是：

- forward network：\(x\rightarrow y\)
- feedback network：\(y\rightarrow x\)
- reverse branch 是 training-time credit assignment / local learning mechanism
- 官方分类代码 validation/test 不运行 reverse network
- forward/backward states通过 local losses 对齐，而不是在 inference state transition 中直接交换

因此我们的主模型必须与此区分。

主模型定义：

\[
z^t = P(H_L^t)
\]

然后：

\[
\boxed{
C_L^{t+1}=B(z^t)
}
\]

其中：

- \(P\)：把 forward terminal representation 映射为当前 output hypothesis；
- \(B\)：把 hypothesis lift 到 reverse boundary latent space。

这叫：

\[
\boxed{\textbf{endogenous target boundary}}
\]

或：

\[
\boxed{\textbf{self-generated terminal boundary}}
\]

GT 只进入最终 task loss：

\[
L_{\rm task}(\hat y,y)
\]

而不直接进入 reverse stream。

---

# 6. 分类任务中的一个具体 boundary 构造

假设有 \(K\) 个类别。

学习 target prototypes：

\[
z_1,\dots,z_K.
\]

当前分类概率：

\[
p^t=\operatorname{softmax}(P(H_L^t)).
\]

构造 target-side hypothesis：

\[
z_{\rm hyp}^t
=
\sum_{k=1}^{K}p_k^t z_k.
\]

然后：

\[
\boxed{
C_L^{t+1}
=
B(z_{\rm hyp}^t)
}
\]

不要第一版使用：

\[
z_{\arg\max p}
\]

因为 hard argmax 容易产生错误 hypothesis 自我强化。

soft hypothesis 保留 uncertainty。

---

# 7. 每个 inference iteration 都必须重新注入原始输入

这是当前设计的硬约束：

\[
\boxed{
H_0^t = E_x(x),\qquad \forall t.
}
\]

也就是说，source evidence 是每轮都被 hard-clamp 的 boundary condition。

第二轮以后不能让网络只围绕 reverse state 自循环。

正确逻辑：

```text
fixed source evidence                      dynamic target hypothesis
        x                                          z^t
        |                                           |
        v                                           v
       H0 -> H1 -> H2 -> ... -> HL
             ↕      ↕            ↕
       C0 <- C1 <- C2 <- ... <- CL
```

每一轮：

- \(x\) 始终重新进入 \(H_0\)；
- 当前 \(H_L\) 生成下一轮 \(C_L\)；
- 中间 H/C 状态重新 reconcile。

这防止系统脱离真实 evidence，变成纯 confirmation loop。

---

# 8. inference 的核心循环

## Iteration 0：纯 forward 初始化

\[
H_0^0=E_x(x)
\]

\[
H_{l+1}^0=F_l(H_l^0)
\]

得到：

\[
z^0=P(H_L^0).
\]

---

## Iteration \(t+1\)

重新 clamp：

\[
H_0^{t+1}=E_x(x).
\]

构造 target boundary：

\[
C_L^{t+1}=B(z^t).
\]

然后所有 layer 做 coupled counterflow update：

\[
(H^{t+1},C^{t+1})
=
\Phi_\theta(H^t,C^t;E_x(x),B(z^t)).
\]

最后：

\[
z^{t+1}=P(H_L^{t+1}).
\]

形成：

\[
\boxed{
x
\rightarrow H_L^t
\rightarrow z^t
\rightarrow C_L^{t+1}
\rightarrow C^{t+1}
\rightarrow H^{t+1}
\rightarrow H_L^{t+1}
}
\]

这是：

\[
\boxed{\textbf{closed-loop counterflow inference}}
\]

---

# 9. 它不是“先入为主”式单向 top-down forcing

存在一个重要风险：

\[
H_L^t\rightarrow C_L^t
\]

可能导致错误 hypothesis 自我强化。

例如：

\[
P(\text{dog})=0.55
\]

然后 dog-like reverse state 强迫 forward representation 更像 dog。

因此 reverse hypothesis 不能单向支配 forward evidence。

本项目要求：

\[
\boxed{
\text{hypothesis corrects evidence}
\quad\text{and}\quad
\text{evidence corrects hypothesis}
}
\]

数学上：

\[
\frac{\partial H_{l+1}}{\partial C_{l+1}}\neq0
\]

并且：

\[
\frac{\partial C_l}{\partial H_l}\neq0.
\]

也就是说：

\[
\boxed{\textbf{reciprocal hypothesis correction}}
\]

而不是普通 top-down feedback。

---

# 10. 与 CCL 最本质的区别

## CCL

forward state：

\[
a_l
\]

reverse state：

\[
b_l
\]

通过：

\[
L(a_l,b_l)
\]

在训练 loss 层面耦合。

这是：

\[
\boxed{\text{loss-level coupling}}
\]

且 classification inference：

\[
\hat y=F_{\rm fw}(x)
\]

不需要 backward branch。

---

## 本项目

forward state：

\[
H_l
\]

reverse state：

\[
C_l
\]

直接进入同一个 layer 的 state transition：

\[
(H_{l+1},C_l)
=
\mathcal E_l(H_l,C_{l+1}).
\]

这是：

\[
\boxed{\text{state-level coupling}}
\]

reverse stream 是 predictor 的组成部分。

因此：

\[
\boxed{
\text{CCL uses counter-current geometry for learning;}
}
\]

而我们研究：

\[
\boxed{
\text{counter-current geometry as inference computation.}
}
\]

---

# 11. 真正的局部 Counterflow Exchange Operator

第 \(l\) 个 counterflow cell 的两个 inlet：

\[
H_l
\]

和：

\[
C_{l+1}.
\]

先做 in-stream transport：

\[
\tilde H_{l+1}
=
F_l(H_l;U_l)
\]

\[
\tilde C_l
=
G_l(C_{l+1};V_l).
\]

然后定义 local discrepancy：

\[
\boxed{
D_l
=
\tilde H_{l+1}
-
\tilde C_l
}
\]

或者更一般在共同 exchange space：

\[
D_l
=
\phi_H(\tilde H_{l+1})
-
\phi_C(\tilde C_l).
\]

---

# 12. exchange flux

定义：

\[
\boxed{
Q_l
=
K_l D_l
}
\]

其中：

\[
K_l\succeq0.
\]

第一版推荐最简单的 channel-wise conductance：

\[
\boxed{
Q_l
=
\gamma_l
\odot
D_l
}
\]

且：

\[
0<\gamma_{l,c}<0.5.
\]

例如：

\[
\gamma_l
=
0.49\sigma(a_l).
\]

这样：

- discrepancy 决定 exchange 方向；
- \(\gamma\) 只决定 conductivity；
- 不使用 arbitrary attention；
- 不使用任意 \(1\times1\) Conv 去改变符号和方向。

---

# 13. paired / antisymmetric state exchange

局部 exchanger：

\[
\boxed{
H_{l+1}
=
\tilde H_{l+1}
-
Q_l
}
\]

\[
\boxed{
C_l
=
\tilde C_l
+
Q_l
}
\]

因此在 exchange substep 内：

\[
H_{l+1}+C_l
=
\tilde H_{l+1}+\tilde C_l.
\]

注意：

> 这不是整个网络的物理 energy conservation。

准确说法：

- paired flux
- antisymmetric exchange
- zero-sum exchange substep

不要 headline 为“能量守恒”。

---

# 14. 与普通 feedback / feature fusion 的区别

不允许把主模型写成：

\[
H'=H+MLP(C)
\]

或：

\[
H'=\operatorname{Attn}(H,C).
\]

因为这样只是 arbitrary fusion。

真正 Counterflow primitive 是：

\[
\boxed{
D_l
\rightarrow
Q_l
\rightarrow
(-Q_l,+Q_l)
}
\]

即：

\[
\boxed{\text{discrepancy-driven paired flux}}
\]

。

---

# 15. “信息熵”只作为后续可验证解释，不作为第一性定义

一个很直观的解释是：

Forward：

\[
\text{high task uncertainty}
\rightarrow
\text{low task uncertainty}
\]

Reverse：

从 target-side low-uncertainty hypothesis 向 source side 展开，逐渐面对更多 underdetermined representation。

但不能直接把任意 hidden feature 的 Shannon entropy 当“温度”。

更严谨的实验定义可使用：

\[
\mathcal H_l
=
-\sum_k p_l(k)\log p_l(k)
\]

即 predictive entropy。

因此：

\[
\boxed{
\text{task uncertainty / representation potential}
}
\]

优先于“信息熵”作为理论术语。

---

# 16. Counterflow vs Co-current 的核心理论动机

理想 paired exchange 下：

## Co-current

\[
h_{l+1}=h_l-q_l
\]

\[
c_{l+1}=c_l+q_l
\]

若：

\[
q_l=\gamma(h_l-c_l)
\]

则 discrepancy：

\[
d_l=h_l-c_l
\]

满足：

\[
\boxed{
d_{l+1}
=
(1-2\gamma)d_l.
}
\]

所以：

\[
|d_l|
\]

随 depth 衰减。

---

## Countercurrent

\[
h_{l+1}=h_l-q_l
\]

\[
c_l=c_{l+1}+q_l.
\]

则在理想 pure exchange 下：

\[
\boxed{
d_{l+1}=d_l.
}
\]

所以 opposing flow 可以在 depth 上维持 distributed discrepancy / driving force。

这条理论不能直接套到任意 nonlinear CNN transport 上，但它给出明确设计原则和实验假说：

\[
\boxed{
\text{Counterflow should preserve usable reciprocal correction deeper into the network better than coflow.}
}
\]

---

# 17. 每层 exchange 还可以形成局部 exchange energy

在线性近似：

\[
\tilde H=UH
\]

\[
\tilde C=VC
\]

定义：

\[
D=UH-VC.
\]

局部 exchange energy：

\[
\boxed{
E_l
=
\frac12
D^\top K D
}
\]

其中：

\[
K\succeq0.
\]

于是：

\[
\boxed{
Q
=
\nabla_D E_l
=
KD.
}
\]

因此同一个 \(Q\)：

- inference 时：state exchange flux；
- learning theory 中：局部 exchange energy 对 discrepancy 的梯度。

这个 dual meaning 是后续理论的重要方向。

---

# 18. Flux-driven local plasticity：高级研究方向，不作为第一版主训练方法

因为：

\[
D=UH-VC
\]

有：

\[
\frac{\partial E_l}{\partial U}
=
QH^\top
\]

以及：

\[
\frac{\partial E_l}{\partial V}
=
-QC^\top.
\]

所以局部 gradient descent：

\[
\boxed{
\Delta U
=
-\eta QH^\top
}
\]

\[
\boxed{
\Delta V
=
+\eta QC^\top
}
\]

这意味着同一个 flux \(Q\) 理论上可以同时控制：

1. 快时间尺度：activation/state exchange；
2. 慢时间尺度：local weight plasticity。

这是非常有潜力的后续问题：

> **Can the same local flux drive both inference and learning?**

但是第一版主模型仍应：

\[
\boxed{\text{正常 end-to-end BP}}
\]

训练，以避免同时引入新 architecture + 新 learning rule。

---

# 19. 三个时间/尺度概念

当前系统可以区分：

### Depth coordinate \(l\)

信息沿网络层位置传播。

### Inference iteration \(t\)

H/C 不断重新交换和 reconcile。

### Training step \(\tau\)

参数 \(U,V,K\) 在数据训练过程中缓慢变化。

它们分别是：

\[
l,\quad t,\quad \tau.
\]

不要把它们混成物理时间。

---

# 20. Boundary-driven non-equilibrium 的解释

每个 local cell 都在根据 discrepancy 发生 exchange。

如果局部独立运行，它会趋向：

\[
D_l\rightarrow0.
\]

但是整个系统两端持续受到不同 boundary 驱动：

\[
H_0^t=E_x(x)
\]

以及：

\[
C_L^t=B(z^t).
\]

因此全局不必 collapse 到同一状态。

这与 countercurrent exchanger 的重要结构相似：

> local interaction tends to reduce discrepancy, while opposing boundaries maintain a distributed gradient.

因此可把系统解释成：

\[
\boxed{\textbf{boundary-driven non-equilibrium neural inference}}
\]

---

# 21. 当前主模型的四种 boundary 版本

## A. Null Boundary

\[
C_L=0.
\]

用途：

- pure topology ablation；
- 检查 reverse flow 没有额外信息时是否仍有价值。

不再作为 conceptual main model。

---

## B. Learned Reservoir

\[
C_L=Z.
\]

Z 是训练参数，所有样本共享。

用途：

- learned task-side prior ablation。

---

## C. Self-Generated Target Boundary —— 主模型

\[
\boxed{
C_L^{t+1}
=
B(P(H_L^t)).
}
\]

或者分类 prototype 版本：

\[
C_L^{t+1}
=
B\left(\sum_kp_k^tz_k\right).
\]

训练、推理都不需要 GT 进入 reverse stream。

---

## D. GT Oracle Boundary

\[
C_L=B(y_{\rm GT}).
\]

只作为：

- oracle upper bound；
- training-only ablation；
- 与 CCL-like paradigm 对比。

不能作为主 inference 结果。

---

# 22. Co-current control 必须使用相同 boundary content

如果 Countercurrent 使用：

\[
C_L=B(z^t)
\]

则 Co-current 不能使用 0。

Co-current 应使用同样的 target hypothesis：

\[
C_0=B(z^t).
\]

唯一主要区别：

Countercurrent：

\[
C:L\rightarrow0.
\]

Co-current：

\[
C:0\rightarrow L.
\]

这样真正测试：

\[
\boxed{
\text{same source + same target hypothesis + same exchange law:
does opposing transport matter?}
}
\]

---

# 23. 第一篇论文最强的 novelty hierarchy

## Core novelty 1 — Endogenous target-side boundary

\[
C_L=B(P(H_L)).
\]

reverse boundary 由模型 inference 自己产生，不依赖 GT。

---

## Core novelty 2 — State-level reciprocal inference

H/C 在 state transition 中直接耦合，而不是只通过 loss coupling。

---

## Core novelty 3 — Discrepancy-driven paired exchange

\[
Q=KD
\]

\[
\Delta H=-Q,\qquad \Delta C=+Q.
\]

---

## Core novelty 4 — Counterflow vs coflow driving-force theory

ideal exchange 下 opposing transport 对 distributed discrepancy 有结构优势。

---

## Advanced novelty 5 — Flux-driven plasticity

\[
Q
\]

同时驱动 state exchange 与 local weight update。

这一点风险更高，建议后续单独研究或作为扩展。

---

# 24. 与主要邻居的定位

## Counter-Current Learning (NeurIPS 2024)

- reverse branch 主要是 training mechanism；
- classification test 不需要 reverse；
- local loss-level coupling；
- target/label-conditioned backward representations。

我们：

- reverse 是 inference computation；
- boundary endogenous；
- state-level exchange；
- iterative closed loop。

---

## Predictive Coding / Feedback Networks

它们已经有：

- top-down feedback；
- recurrent refinement；
- iterative inference。

因此：

> “test-time feedback” 本身不是 strong novelty。

我们的强区别必须来自：

- independently transported opposing fields；
- explicit paired flux；
- counter-vs-co structural theory；
- endogenous target boundary。

---

# 25. 第一阶段不要做的事情

主实验第一版不要：

- 把 GT 直接送进 reverse stream；
- 用 hard argmax target boundary；
- 去掉每轮输入 \(x\) 的 hard clamp；
- 把 reverse 变成普通 decoder；
- 只在最终 concat H/C；
- 用 arbitrary Cross-Attention 替代 paired flux；
- 同时取消 Backprop；
- 同时做 DEQ；
- 宣称 time-reversal symmetry；
- 宣称整个网络 energy conservation；
- 直接把 hidden feature entropy 当物理熵。

---

# 26. 最终 method intuition

最准确的人类直觉是：

> **网络不仅从 evidence 预测 target，还利用当前 target hypothesis 重新解释 evidence；同时 evidence 也反过来修正 hypothesis。**

不是：

\[
\text{hypothesis}\rightarrow\text{force evidence}
\]

而是：

\[
\boxed{
\text{hypothesis}
\leftrightarrow
\text{evidence}
}
\]

最终目标：

\[
\boxed{
\textbf{iterative hypothesis–evidence reconciliation under counterflow exchange}
}
\]

---

# 27. 实现 Agent 必须牢记

真正主模型必须保留：

1. source boundary \(H_0=E(x)\)；
2. 每个 iteration 都重新 clamp \(H_0\)；
3. target-side boundary \(C_L\) 由上一轮 \(H_L\) 产生；
4. H/C 反向传播方向相反；
5. 每个 layer 都有真实 state-level exchange；
6. exchange 由 discrepancy 驱动；
7. 同一个 \(Q\) 以 \(-Q,+Q\) 成对进入 H/C；
8. reverse hypothesis 可以被 evidence 修正；
9. output 主要从 forward terminal state得到；
10. Co-current control 与 Countercurrent 使用完全相同的 boundary content；
11. GT 不进入主模型 reverse stream；
12. normal BP 作为第一阶段训练方式。

不要为了方便把它改成：

```text
forward CNN -> output -> backward decoder -> concat -> classifier
```

那不是本项目。
