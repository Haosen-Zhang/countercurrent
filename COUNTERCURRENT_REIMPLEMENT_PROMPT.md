请重新阅读项目中的以下两个 Markdown 文件，并以它们作为最高优先级设计规范：

1. COUNTERCURRENT_CORE_IDEA.md
2. COUNTERCURRENT_CNN_POC_EXPERIMENT.md

当前仓库里之前实现的 Countercurrent 网络是基于一个错误的逆流换热理解写出来的，因此不要在旧实现上做局部 patch。请先审计旧代码，然后按照两个 Markdown 中的最新定义重构网络。如果旧代码的结构与最新定义冲突，应删除/替换旧逻辑，而不是为了兼容旧实现保留错误结构。

========================
一、必须先纠正的物理/结构定义
========================

之前错误理解类似：

hot  -----------------> cold
cold <----------------- hot

这不是真正的 countercurrent heat exchange。

正确的逆流换热应该是：

在共同的左→右空间坐标上：

Hot stream:
HOT inlet  -----------------------------> LESS HOT outlet
                    flow →

Cold stream:
LESS COLD / WARMER outlet <------------- COLD inlet
                          ← flow

也就是说：

1. hot stream 从左侧进入，沿自己的流向逐渐降温；
2. cold stream 从右侧进入，沿自己的流向（右→左）逐渐升温；
3. 两条流在共同空间坐标上具有相似的总体状态梯度，但传播方向相反；
4. reverse stream 的右侧 inlet 必须是 target-side / cold-side state，
   不能是 source-side / hot-side state。

映射到神经网络：

source side                                  target side

H0  ---------------------------------------> HL
     forward flow →

C0  <--------------------------------------- CL
                          ← reverse flow

H0 来自输入 x：

    H0 = E(x)

CL 位于 output/target 一侧。

主模型中 CL 不使用 GT，而由当前 forward terminal state HL 自己产生：

    z^t = P(HL^t)
    CL^(t+1) = B(z^t)

对于分类问题，可以使用 soft class hypothesis：

    p^t = softmax(Head(HL^t))
    z_hyp^t = sum_k p_k^t * z_k
    CL^(t+1) = B(z_hyp^t)

因此：

CL 是 target-side hypothesis boundary，
不是 source-like state，
也不是为了“反向”而人为给一个和 target 相反的 state。

===================================
二、必须实现的核心 inference 逻辑
===================================

每个 inference iteration t 都必须重新注入原始输入 evidence：

    H0^t = E(x),  for every t

绝对不能只在 t=0 输入一次 x，然后 t>0 让网络只围绕 CL / reverse state 自循环。

整个 inference 应该是：

Iteration 0:
    x -> H0 -> H1 -> ... -> HL
                         |
                         v
                       z^0

Iteration 1:
    x -> H0 -> H1 -> H2 -> ... -> HL
             <->    <->          <->
    C0 <- C1 <- C2 <- ... <-     CL
                                  ^
                                  |
                                B(z^0)

得到 z^1。

Iteration 2:
重新输入同一个 x：

    x -> H0 -> H1 -> H2 -> ... -> HL
             <->    <->          <->
    C0 <- C1 <- C2 <- ... <-     CL
                                  ^
                                  |
                                B(z^1)

依次迭代。

所以核心闭环是：

    x
      -> forward evidence
      -> HL^t
      -> current target hypothesis z^t
      -> CL^(t+1)
      -> reverse target-side representation
      -> layer-wise exchange with forward states
      -> revised forward representation
      -> HL^(t+1)

这不是：
    CNN -> decoder -> concat -> classifier

也不是：
    GT -> reverse network

而是：
    fixed evidence + dynamically inferred target boundary
    之间的 iterative reciprocal reconciliation。

===================================
三、每一层必须发生真正的局部交换
===================================

Countercurrent 不能只靠 loss 让两个网络互相学习。

每个 layer 的 state transition 本身必须包含 forward/reverse interaction。

对于第 l 个 countercurrent cell：

forward inlet:
    H_l

reverse inlet:
    C_{l+1}

先分别 transport：

    H_tilde_{l+1} = F_l(H_l; U_l)

    C_tilde_l = G_l(C_{l+1}; V_l)

定义 discrepancy：

    D_l = H_tilde_{l+1} - C_tilde_l

第一版不要使用 arbitrary Cross-Attention，也不要使用任意 1x1 Conv 改变 discrepancy 方向。

使用 channel-wise positive conductance：

    gamma_l = 0.49 * sigmoid(a_l)

所以：

    0 < gamma_l < 0.5

local exchange flux：

    Q_l = gamma_l ⊙ D_l

然后必须使用同一个 Q_l 做 paired exchange：

    H_{l+1} = H_tilde_{l+1} - Q_l

    C_l     = C_tilde_l     + Q_l

因此：

    ΔH = -Q
    ΔC = +Q

同一个 local flux 同时作用于正流和逆流。

这才是真正想研究的 Counterflow Exchange Operator。

请不要实现成：

    H' = H + MLP(C)

或者：

    H' = Attention(H,C)

也不要分别学习两个完全独立的：

    Q_H
    Q_C

作为主模型。

===================================
四、正逆流不是单向“先入为主”
===================================

target hypothesis 可以改变 forward representation，
但 evidence 也必须能够改变 reverse hypothesis。

我们想实现的是：

    hypothesis <-> evidence

而不是：

    hypothesis -> evidence

也就是说 reverse state 不应该只是强行把 H 拉向当前 prediction。

需要保证：

    ∂H_{l+1} / ∂C_{l+1} != 0

并且：

    ∂C_l / ∂H_l != 0

所以每一层都是 reciprocal state coupling。

这也是为什么使用：

    H' = H_tilde - Q
    C' = C_tilde + Q

而不是只有 C -> H 的单向 feedback。

===================================
五、主模型 boundary 设置
===================================

请实现并清楚区分以下 boundary variants。

A. Self-generated target boundary —— 主模型

    CL^(t+1) = B(P(HL^t))

分类时推荐：

    p = softmax(logits)
    z_hyp = p @ class_prototypes
    CL = boundary_projector(z_hyp)

禁止主版本使用 argmax class prototype。

B. Null boundary —— ablation

    CL = 0

它现在只是 topology ablation，不是主模型。

C. Learned-Z boundary —— ablation

    CL = learned parameter Z

所有样本共享。

D. GT Oracle boundary —— 仅分析

    CL = B(y_GT)

只能作为 oracle / upper-bound / CCL-like comparison。

主模型 forward(x) 不允许接收 label 参数。

===================================
六、Co-current baseline 必须重新实现正确
===================================

Countercurrent：

    H0 = E(x)
    H : 0 -> L

    CL = B(z)
    C : L -> 0

Co-current 必须使用完全相同的 target hypothesis、参数规模和 exchange law，只改变 reverse stream 的 inlet 与传播方向：

    H0 = E(x)
    H : 0 -> L

    C0 = B(z)
    C : 0 -> L

也就是说 Countercurrent vs Co-current 的主要变量只能是：

    opposing transport
vs
    same-direction transport

不能 Countercurrent 用 self-generated target，而 Co-current 用 zero boundary。

===================================
七、每轮必须重新输入 x
===================================

这是这次修改中最重要的一条之一。

每个 refinement iteration：

    source = stem(x)

必须重新计算/重新 clamp：

    H0^t = source

不能让：

    H0^(t+1) = H0^t + reverse feedback

也不能在第二轮以后让网络失去 raw/source evidence anchor。

直觉：

每一轮不是“只根据上一轮猜测继续幻想”，
而是“带着上一轮 hypothesis 再重新看一次同一个输入”。

这也是防止 confirmation bias 的关键设计。

===================================
八、初始实现建议
===================================

按照 COUNTERCURRENT_CNN_POC_EXPERIMENT.md 实现：

Dataset:
    CIFAR-10

channels:
    64

lattice depth:
    L = 4

refinement steps:
    T = 3 initially

state shape:
    [B, 64, 8, 8]

normalization:
    GroupNorm

activation:
    SiLU

transport:
    residual CNN block

exchange:
    Q = gamma ⊙ D
    0 < gamma < 0.5

training:
    normal end-to-end backpropagation

loss:
    final cross entropy only initially

不要现在实现：
    DEQ
    target propagation
    no-backprop learning
    flux-driven weight plasticity
    cross-attention
    Transformer
    U-Net
    multi-resolution reverse decoder

这些是后续实验。

===================================
九、必须实现的 baselines
===================================

至少保留：

1. SingleStream-FeedForward

2. Forward-Only Recurrent Refinement
   控制“多迭代几次”本身的收益。

3. Feedback-Fusion baseline
   有 top-down feedback，但没有 paired counterflow flux。

4. Co-current + self-generated target boundary

5. Countercurrent + self-generated target boundary

6. Countercurrent reverse-off evaluation

7. Null-boundary Countercurrent

8. Learned-Z Countercurrent

如果现有代码里已有旧版本 Countercurrent，不要把它覆盖后完全删除历史结果；
可以重命名为 legacy/old_wrong_countercurrent，仅用于回溯，
但不要作为新实验的主模型。

===================================
十、必须记录的 diagnostics
===================================

每个 refinement iteration t：

    accuracy(t)
    logits(t)
    confidence(t)
    predictive entropy(t)

每个 layer l：

    ||D_l||
    ||Q_l||
    ||H_l||
    ||C_l||
    gamma_l

另外实现：

1. Reverse-Off Test

    正常：
        Q != 0

    测试关闭：
        Q = 0

    看 accuracy 是否明显下降。

2. Error Correction Rate

    initial prediction wrong
    final prediction correct

3. Error Amplification Rate

    initial prediction wrong
    final wrong-class confidence 更高

这用于判断 self-generated hypothesis 是帮助纠错，
还是造成 confirmation bias。

===================================
十一、必须注意 CCL 的区别
===================================

不要把本模型写成 CCL 的简单变体。

CCL 的关键是：

    forward x -> y
    backward label/class -> x

两者主要通过 local training loss 发生 coupling。

classification test 时 reverse branch 不参与 prediction。

我们这里：

    reverse branch 是 inference function 的组成部分

并且：

    CL 来自模型自身当前 HL / output hypothesis

而不是 GT。

更重要的是：

    H 和 C 在每个 layer 的 state transition 内通过 Q_l 直接交换。

所以本项目的关键 distinction 是：

    CCL:
        loss-level coupling
        training-time reverse pathway

    Ours:
        state-level coupling
        inference-time reverse pathway
        endogenous target boundary
        discrepancy-driven paired flux
        closed-loop iterative inference

===================================
十二、完成修改后请输出
===================================

完成实现后，不要直接启动长时间完整训练。

请先输出：

1. 旧模型和新模型 architecture diff；
2. 哪些旧实现属于错误的 countercurrent mapping；
3. 新模型 forward() 的完整 data flow；
4. Countercurrent / Co-current 的 ASCII diagram；
5. 参数量；
6. FLOPs；
7. unit tests 结果；
8. 一个 batch 的 forward sanity check；
9. 每个 iteration 的 tensor shapes；
10. gamma / D / Q 的初始统计；
11. 确认主模型完全没有 GT leakage；
12. 确认每个 refinement iteration 都重新注入 x。

然后先运行小规模 debug：

    1-5 epochs

确认：
    loss 正常下降
    没有 NaN/Inf
    Q 不全为 0
    gamma 合法
    H/C norm 不爆炸
    reverse-off 可以正常运行

确认无误后，再进入正式 CIFAR-10 实验。

最重要的一句话：

不要把旧版“两个方向相反的网络”直接当成真正 Countercurrent。

真正的新版本必须体现：

    target-side reverse inlet
    +
    opposite transport
    +
    local paired exchange at every layer
    +
    fixed source evidence injected every iteration
    +
    self-generated target hypothesis

也就是：

    每一个 layer 都像真正逆流换热器的一个局部 exchanger cell。
