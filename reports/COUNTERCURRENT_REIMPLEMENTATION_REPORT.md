Countercurrent reimplementation audit and validation — 2026-09-14

The active network now uses the proposal/reconciliation/forward-resweep schedule
from POC section 15, with functional evidence-to-reverse correction. Both
reconciliation stages use paired flux. All 34 unit tests pass, and a default-size
network passed one-batch forward, backward, one optimizer update, and reverse-off
checks. No epoch training or full CIFAR experiment was started.

The authoritative inputs were read before auditing:
[reimplementation requirements](../COUNTERCURRENT_REIMPLEMENT_PROMPT.md),
[core idea](../COUNTERCURRENT_CORE_IDEA.md), and
[CNN POC specification](../COUNTERCURRENT_CNN_POC_EXPERIMENT.md).

The available pre-edit snapshot was already partially migrated: its main boundary
was a soft self-generated hypothesis, its stem was called at every iteration, and
its conductance was bounded. It would be inaccurate to attribute a source-like
reverse inlet or GT leakage to those files. The earlier implementation described
in the prompt was not present as a separate version, and Git history was not
available in this workspace. This audit concerns the code actually present.

The remaining conceptual defect was in `CoupledLatticeBase._run`: every iteration
computed `corrected_c[c_index] = c_ref + q`, but prediction consumed only
`c_states[c_index]`, the uncorrected proposal. The corrected C values were copied
into diagnostics and discarded. Thus opposite transport and a displayed `+Q`
were being treated as sufficient evidence of reciprocal computation. In the
predictor, that positive flux branch had no effect. The original tests checked
local exchange algebra but never checked whether the network consumed corrected C;
all 23 original tests passed before the change.

| Aspect | Available old implementation | Active implementation |
|---|---|---|
| Source | `stem(x)` at t=0 and each refinement | Retained and explicitly tested |
| Target inlet | Soft hypothesis at C_L; Co at C_0 | Retained; shared parameterization and boundary content |
| Transport schedule | Reverse proposal, then corrected forward | Forward/reverse proposals, paired reconciliation, corrected forward resweep |
| Evidence → reverse | `C + Q` only recorded | Reconciled `C + Q` is a live resweep input at every layer |
| Flux | One H-active exchange per cell | Paired proposal exchange and paired resweep exchange, each using one shared Q for its H/C pair |
| Output | GAP(H_L) → Linear | Retained; no H/C concatenation |
| Forward recurrent | F/R passes, matched old transport budget | Additional evidence proposal pass, matched new transport budget |
| GT oracle | Missing | Separate analysis-only wrapper, outside main registry |
| Diagnostics | Final H/C and scalar D/Q norms | Both stages' tensors/norms, initial/refined predictions, confidence/entropy |
| Historical loading | Same shapes could silently imply old semantics | Archived models/configs; new inference-version buffer rejects old strict checkpoint loads |

The [legacy snapshot](../countercurrent_nn/legacy/old_wrong_countercurrent/README.md)
preserves the original models, configs, README and SHA-256 manifest. Existing
result files were left in place; none were present beyond the results README.
New schedule configs use `_v2_seed0` output directories. Reusable components that
already complied with the specification—stem, residual transport, boundary and
conductance—were retained. The obsolete update schedule was replaced.

The full forward flow is implemented in
[coupled.py](../countercurrent_nn/models/coupled.py):

```text
t = 0:
    source = stem(x)
    H[0] = source
    H[l+1] = F[l](H[l]), l = 0..L-1
    logits(0) = head(mean_spatial(H[L]))

For t = 1..T:
    source = stem(x)                       # recompute from the same original x
    p = softmax(logits(t-1))               # no detach, no argmax, no GT
    z = p @ class_prototypes
    boundary = broadcast(projector(z))    # [B,64,8,8]

    H_prop[0] = source
    H_prop[l+1] = F[l](H_prop[l])          # left-to-right proposal

    Countercurrent:
        C_prop[L] = boundary
        C_prop[l] = G[l](C_prop[l+1])      # l = L-1..0
        reverse reference index j(l) = l
    Co-current:
        C_prop[0] = boundary
        C_prop[l+1] = G[l](C_prop[l])      # l = 0..L-1
        reverse reference index j(l) = l+1

    For each cell l, reconcile the proposals:
        D_prop = H_prop[l+1] - C_prop[j(l)]
        Q_prop = gamma[l] * D_prop
        H_rec[l+1] = H_prop[l+1] - Q_prop
        C_rec[j(l)] = C_prop[j(l)] + Q_prop

    H_new[0] = source                      # hard clamp again
    For l = 0..L-1, corrected forward resweep:
        h_bar = F[l](H_new[l])
        D = h_bar - C_rec[j(l)]            # consumes the corrected C, not C_prop
        Q = gamma[l] * D
        H_new[l+1] = h_bar - Q
        C_new[j(l)] = C_rec[j(l)] + Q

    logits(t) = head(mean_spatial(H_new[L]))

Return logits(T). Train with cross_entropy(logits(T), labels) only.
```

`gamma[l] = 0.49 * sigmoid(a[l])`, with `a[l] = -2.2` initially. Both stages share
the same channel-wise conductance parameters. Their fluxes differ because their
discrepancies differ; neither stage learns independent Q_H/Q_C. The proposal
reconciliation's C output is consumed by the resweep; its H output is superseded
by that resweep, as in POC section 15. The final C output is recorded, and the next
iteration rebuilds proposals from x and the revised logits. This is a finite
sweep approximation, not an equilibrium solution or a Jacobi/DEQ lattice. In
particular, the reverse proposal transport runs before paired reconciliation;
corrected C is then consumed by the forward resweep, not retransported by G in
the same iteration.

The shortened POC section 17 pseudocode makes the paired C update optional and
does not consume it. Following it literally would retain the audited defect.
The chosen section 15 schedule resolves that inconsistency in favor of the core
specification's required reciprocal influence. The extra resweep flux is paired
as well, although section 15's abbreviated resweep only writes its H update.

```text
Countercurrent:        source side, depth 0                 target side, depth L

    x -> E(x) -> H0 --------> H1 --------> ... --------> HL -> head -> logits(t)
                 C0 <-------- C1 <-------- ... <-------- CL
                                                        ^
                                            B(soft hypothesis from logits(t-1))

    Cell l proposal pair: F_l(H_l) exchanges with G_l(C_{l+1}).
    Opposing transport: H 0 -> L, C L -> 0.

Co-current:            depth 0                              depth L

    x -> E(x) -> H0 --------> H1 --------> ... --------> HL -> head -> logits(t)
                 C0 --------> C1 --------> ... --------> CL
                 ^
       B(soft hypothesis from logits(t-1))

    Cell l proposal pair: F_l(H_l) exchanges with G_l(C_l).
    Same-direction transport: H 0 -> L, C 0 -> L.
```

Both controls use identical modules, independent F/G weights, the same soft
boundary generator, number of sweeps, head and exchange law. Shared state dicts
load across the two controls. After the first refinement their hypothesis values
may differ because topology changes predictions; the boundary rule remains the
same. Tests check identical initial boundary content under identical weights.

Default stem: 3×3 Conv/GN/SiLU with strides 1,2,2, width 64. Each transport is
`x + 0.1 * Conv(SiLU(GN(Conv(SiLU(GN(x))))))`, using two 3×3 convolutions.
All lattice tensors remain at 8×8; GroupNorm has eight groups. L=4, T=3,
prototype dimension=64, classes=10. F/G weights are independent and reused over
iterations. There is no attention, decoder, multi-resolution reverse path, local
energy loss, non-backprop optimizer, or final H/C feature fusion in the main model.

Measured parameter and compute counts, batch=1, input `[1,3,32,32]`:

| Model | Trainable parameters | Conv/Linear MACs | Approx. FLOPs: 2 × counted MACs |
|---|---:|---:|---:|
| Old Countercurrent / Co-current | 673,418 | 186,399,232 | 372,798,464 |
| Single feed-forward, L=4 | 372,426 | 32,440,960 | 64,881,920 |
| Single feed-forward, L=8, approximate parameter match | 668,362 | 51,315,328 | 102,630,656 |
| Forward-only recurrent refinement | 668,362 | 243,010,048 | 486,020,096 |
| Feedback fusion | 689,802 | 189,544,960 | 379,089,920 |
| Co-current, self-generated boundary | 673,418 | 243,022,336 | 486,044,672 |
| Countercurrent, self-generated boundary | 673,418 | 243,022,336 | 486,044,672 |
| Countercurrent, reverse-off evaluation | 673,418 | 243,022,336 | 486,044,672 |
| Countercurrent, null boundary | 668,618 | 243,010,048 | 486,020,096 |
| Countercurrent, learned-Z boundary | 672,714 | 243,010,048 | 486,020,096 |

The new main-model count includes four stem calls, 28 F-block calls, 12 G-block
calls, four head calls and three boundary-projector calls. The new proposal pass
adds 56,623,104 MACs without adding trainable parameters. Reverse-off still runs
transport and zeroes both exchange fluxes; it is an intervention, not a speed
optimization. The forward recurrent model matches transport compute; its missing
projector accounts for the 12,288 counted MAC difference. The L=8 single stream
has 0.75% fewer parameters than the main model. Feedback fusion is a qualitative
feedback control with the lower compute shown above, not an equal-compute claim.

FLOPs here cover Conv2d/Linear arithmetic only, using two FLOPs per MAC. They
exclude GroupNorm, SiLU, pooling, softmax, biases, and elementwise exchange. The
soft prototype matrix products add 1,920 MACs per image over T=3, also outside
that scope. These are transparent approximate operation counts, not complete
hardware instruction counts. Quick CPU latency measurements are in the JSON;
CPU peak memory was not measured (`peak_memory_bytes=0` is the profiler sentinel).

The one-batch check used seed 0, PyTorch 2.6.0+cu124, CPU, one thread, B=8,
and the default 64-channel L=4/T=3 model. CUDA was unavailable in the current
environment. Input was synthetic uniform RGB, normalized with CIFAR-10 statistics,
with random labels; no real CIFAR batch was loaded. Results are engineering sanity
checks, not CIFAR performance measurements.

| Iteration | Each H state | Each C state | Logits | Batch accuracy | Mean confidence | Predictive entropy |
|---|---|---|---|---:|---:|---:|
| 0 | `[8,64,8,8]`, 5 depths | Not run | `[8,10]` | 0.125 | 0.131023 | 2.285104 |
| 1 | `[8,64,8,8]`, 5 depths | `[8,64,8,8]`, 5 depths | `[8,10]` | 0.125 | 0.127110 | 2.288949 |
| 2 | `[8,64,8,8]`, 5 depths | `[8,64,8,8]`, 5 depths | `[8,10]` | 0.125 | 0.127110 | 2.288949 |
| 3 | `[8,64,8,8]`, 5 depths | `[8,64,8,8]`, 5 depths | `[8,10]` | 0.125 | 0.127110 | 2.288949 |

Input shape was `[8,3,32,32]`. H history is `[4,5,8,64,8,8]`; C history is
`[3,5,8,64,8,8]`. Each stage's D/Q history is `[3,4,8,64,8,8]`. Boundary history
is `[3,8,64,8,8]`; stacked logits are `[4,8,10]`. The stem hook observed exactly
four calls with the original x, and every recorded H0 equaled that iteration's
fresh stem output. Thus evidence is re-injected at every iteration.

Initial conductance was 0.0488777384 in every channel/layer, strictly between zero
and 0.5. Initial per-cell RMS statistics are below; the JSON also contains
min/mean/max and the specified `L2 / number_of_elements` norms for every entry.

| t | Cell l | Proposal D | Proposal Q | Resweep D | Resweep Q | H outlet | C outlet |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.610234 | 0.029827 | 0.580407 | 0.028369 | 0.572950 | 0.097914 |
| 1 | 1 | 0.611460 | 0.029887 | 0.553303 | 0.027044 | 0.546073 | 0.091896 |
| 1 | 2 | 0.610591 | 0.029844 | 0.525601 | 0.025690 | 0.522189 | 0.088992 |
| 1 | 3 | 0.610454 | 0.029838 | 0.499994 | 0.024439 | 0.499804 | 0.079338 |
| 2 | 0 | 0.610239 | 0.029827 | 0.580412 | 0.028369 | 0.572950 | 0.097911 |
| 2 | 1 | 0.611464 | 0.029887 | 0.553306 | 0.027044 | 0.546073 | 0.091890 |
| 2 | 2 | 0.610596 | 0.029845 | 0.525606 | 0.025690 | 0.522189 | 0.088987 |
| 2 | 3 | 0.610460 | 0.029838 | 0.499999 | 0.024439 | 0.499803 | 0.079335 |
| 3 | 0 | 0.610239 | 0.029827 | 0.580412 | 0.028369 | 0.572950 | 0.097911 |
| 3 | 1 | 0.611464 | 0.029887 | 0.553306 | 0.027044 | 0.546073 | 0.091890 |
| 3 | 2 | 0.610596 | 0.029845 | 0.525606 | 0.025690 | 0.522189 | 0.088987 |
| 3 | 3 | 0.610460 | 0.029838 | 0.499999 | 0.024439 | 0.499803 | 0.079335 |

Both stages had nonzero flux at every layer/iteration. H/C states and all
parameter gradients were finite. Final cross-entropy was **2.217976**, falling
to **2.112089** after exactly one AdamW update (lr=3e-4, weight decay=1e-4) on
the same batch. All parameters and states remained finite after the update;
gamma stayed in `[0.04886455, 0.04889095]`. This single update demonstrates a
working optimization path, not multi-epoch convergence.

Before that update, reverse-off made both proposal and resweep Q exactly zero
and reproduced the initial feed-forward logits. Full versus reverse-off logits
differed by up to 0.0588083. Both batch accuracies were 12.5%, so the observed
accuracy drop was zero. Error correction and amplification rates were both zero
among seven initially wrong examples (amplification delta=0.1, tracking the
initial wrong class). Refinements 2 and 3 were almost identical at initialization.
The check establishes computational dependency, not useful learned correction
or an advantage of Countercurrent over Co-current.

No GT enters the main model: `forward(x, *, return_diagnostics=False,
reverse_off=False)` has no label argument, and passing labels positionally or by
keyword raises TypeError. The soft boundary is differentiable with respect to the
previous logits. Final CE reaches the earlier hypothesis, prototypes, projector,
both transport banks, stem, conductance, and head. Labels are used outside forward
for loss and metrics. The optional
[GTOracleCountercurrent](../countercurrent_nn/analysis/oracle_boundary.py) wrapper
explicitly uses one-hot labels and marks its diagnostic output analysis-only;
it is absent from the active registry and does not change normal inference.

Validation commands:

```bash
conda run -n 4dflow python -m unittest discover -s tests -v
conda run -n 4dflow python -m countercurrent_nn.sanity_check --batch-size 8
```

The tests cover shapes, conductance, conservation of each exchange substep,
source re-injection, soft previous-logit boundaries, transport direction, matched
CC/Co parameterization and MACs, both-stage reverse-off, final-only head input,
gradient flow, diagnostics-on/off equivalence, and all active YAML models.
Regression tests additionally prove that every reconciled C is on the prediction
graph and that removing only its positive flux changes predictions with diagnostics
disabled. Oracle isolation and legacy-checkpoint rejection are also tested.

Artifacts: [unit-test log](unit_tests.txt),
[full numerical report](one_batch_sanity.json), and
`reports/one_batch_sanity.pt` containing initial logits and all recorded states,
discrepancies and fluxes. The PT file is a diagnostic artifact, not a trained
checkpoint. Existing analysis scripts can read its standard diagnostic fields.

Work stops at the requested review point. The 1–5 epoch real-CIFAR debug phase
and the full experiment matrix remain unrun. The debug CLI now defaults to three
epochs when explicitly invoked. Scientific conclusions about correction,
confirmation bias, reverse-off accuracy loss or CC-vs-Co require those later
experiments and multiple seeds.
