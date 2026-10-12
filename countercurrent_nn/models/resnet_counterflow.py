"""Standard ResNet-50 with a counterflow lattice in place of its third stage.

Motivation.  Every previous counterflow result in this repository used a bespoke
64-channel 8x8 backbone, which makes the comparison against "a normal network"
unconvincing and leaves the compute accounting unanchored.  This module puts the
same mechanism on a standard, recognisable backbone.

Design
------
The lattice is ResNet-50's third stage.  With the usual CIFAR adaptation (3x3
stride-1 stem, no max-pool) the shapes are

    stem, layer1        32x32 x 256
    layer2              16x16 x 512
    stage_entry          8x8  x 1024     (layer3 block 0, stride 2)
    lattice (5 cells)    8x8  x 1024     (layer3 blocks 1..5, stride 1)
    layer4               4x4  x 2048
    avgpool, fc

so the lattice is five same-shaped cells, which is what the exchange operator
needs, and everything outside it is stock ResNet-50.

Parameter fairness.  The reverse transport is tied to the forward transport:
G_l is the exact inverse of F_l.  That is why the lattice blocks must be
invertible.  Tying costs no parameters and keeps the comparison honest; a
separate learned G would add roughly 5.6M parameters and make any accuracy
difference uninterpretable.  ResNet50LatticeControl keeps the same invertible
stage with the exchange switched off, so the cost of the invertibility
constraint itself is measured rather than assumed.

The target-side boundary needs a class hypothesis at every solver step.  Running
layer4 plus the classifier at every step would multiply the back-end cost by the
step count, so a small linear hypothesis_head reads the lattice terminal state
for that purpose only.  The reported prediction still comes from the stock
layer4 -> avgpool -> fc path, identical to the plain baseline.

Solver.  One sweep is a synchronous Jacobi map: every transport proposal reads
the previous sweep's states, all cells are exchanged simultaneously, then the
result is damped and both boundaries are re-clamped.  Only H_0 (the source
inlet) and C at the target inlet are fixed; the interior persists across sweeps.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck, conv1x1

from .decomposed_boundary import DecomposedTargetBoundary
from .exchange import ChannelwiseConductance
from .invertible_transport import SpectralNormConv2d

Topology = Literal["countercurrent", "cocurrent"]

LATTICE_CHANNELS = 1024
BOTTLENECK_CHANNELS = 256
ENTRY_CHANNELS = 512
EXPANSION = Bottleneck.expansion


class InvertibleBottleneck(nn.Module):
    """ResNet bottleneck whose residual branch is a guaranteed contraction.

    torchvision's Bottleneck ends in BatchNorm and a ReLU, so it is not
    invertible.  This variant keeps the 1x1 -> 3x3 -> 1x1 shape but uses
    GroupNorm (batch-independent, so the forward map is well defined in train
    mode), drops the trailing activation, bounds the residual branch with a
    spectral norm on the last convolution and scales it by beta.  The block is
    then x + beta * f(x) with Lip(beta * f) <= beta, so its inverse is a
    convergent fixed-point iteration.
    """

    expansion = EXPANSION

    def __init__(
        self,
        channels: int = LATTICE_CHANNELS,
        bottleneck_channels: int = BOTTLENECK_CHANNELS,
        *,
        num_groups: int = 32,
        beta: float = 0.1,
        inverse_steps: int = 15,
        inverse_tol: float = 1e-7,
    ) -> None:
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channel counts must be positive")
        if channels % num_groups or bottleneck_channels % num_groups:
            raise ValueError("num_groups must divide both channel counts")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1) for the inverse to converge")
        if inverse_steps < 1:
            raise ValueError("inverse_steps must be at least 1")
        self.beta = float(beta)
        self.inverse_steps = int(inverse_steps)
        self.inverse_tol = float(inverse_tol)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, 1, bias=False),
            nn.GroupNorm(num_groups, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(
                bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups, bottleneck_channels),
            nn.SiLU(),
            SpectralNormConv2d(
                bottleneck_channels, channels, kernel_size=1, padding=0, target=1.0
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.beta * self.residual(x)

    def inverse(self, y: Tensor) -> Tensor:
        """Solve x = y - beta * f(x) by fixed-point iteration.

        The converged solution is obtained without gradient tracking and then
        re-expressed as y - beta * f(x*), so the result is differentiable in the
        input with the exactly correct identity term.  This mirrors
        InvertibleTransportBlock.inverse.
        """
        with torch.no_grad():
            x = y
            for _ in range(self.inverse_steps):
                previous = x
                x = y - self.beta * self.residual(x)
                if (x - previous).norm() <= self.inverse_tol * (
                    previous.norm() + self.inverse_tol
                ):
                    break
        return y - self.beta * self.residual(x.detach())

    def extra_repr(self) -> str:
        return f"beta={self.beta}, inverse_steps={self.inverse_steps}"


class ExactInverseBottleneck(nn.Module):
    """Adapter exposing a bottleneck's exact inverse as the reverse transport.

    Holds no parameter of its own: every tensor it reports belongs to the
    forward block, which is what makes the tied comparison parameter-neutral.
    """

    def __init__(self, block: InvertibleBottleneck) -> None:
        super().__init__()
        self.block = block

    def forward(self, inlet: Tensor) -> Tensor:
        return self.block.inverse(inlet)


def _stage_entry() -> Bottleneck:
    """layer3 block 0: 16x16x512 -> 8x8x1024, kept as stock torchvision."""
    downsample = nn.Sequential(
        conv1x1(ENTRY_CHANNELS, LATTICE_CHANNELS, stride=2),
        nn.BatchNorm2d(LATTICE_CHANNELS),
    )
    return Bottleneck(
        ENTRY_CHANNELS, BOTTLENECK_CHANNELS, stride=2, downsample=downsample
    )


def _make_stage(
    in_channels: int, planes: int, *, blocks: int, stride: int
) -> nn.Sequential:
    out_channels = planes * EXPANSION
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
        )
    layers: list[nn.Module] = [Bottleneck(in_channels, planes, stride, downsample)]
    layers.extend(Bottleneck(out_channels, planes) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class LightweightTargetBoundary(nn.Module):
    """Decomposed target boundary sized for a wide backbone.

    The V5-C boundary is comfortable at 64 channels but at 1024 channels its
    instance encoder alone is 10.5M parameters, adding 54% on top of ResNet-50.
    That would make every accuracy comparison uninterpretable.  This keeps the
    same three components -- learned base, centered class map, stop-gradient
    instance -- but uses a low-rank class projection followed by a depthwise
    spatial conv, and a separable instance encoder.  Total overhead is about
    0.37M, roughly 1.6% of ResNet-50.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        channels: int,
        spatial_size: int = 8,
        class_rank: int = 32,
        instance_hidden: int = 128,
        num_groups: int = 32,
        base_scale: float = 0.15,
        class_scale: float = 1.0,
        instance_scale: float = 0.1,
        target_rms: float | None = 0.55,
    ) -> None:
        super().__init__()
        if min(num_classes, channels, spatial_size, class_rank, instance_hidden) <= 0:
            raise ValueError("boundary dimensions must be positive")
        if channels % num_groups or instance_hidden % num_groups:
            raise ValueError("num_groups must divide both channel counts")
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.base_state = nn.Parameter(
            torch.randn(1, channels, spatial_size, spatial_size)
        )
        self.class_embedding = nn.Parameter(
            torch.randn(num_classes, class_rank) / math.sqrt(class_rank)
        )
        self.class_channel = nn.Linear(class_rank, channels, bias=False)
        # depthwise 3x3 turns the per-channel class response into a spatial map
        self.class_spatial = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.instance_encoder = nn.Sequential(
            nn.Conv2d(channels, instance_hidden, 1, bias=False),
            nn.GroupNorm(num_groups, instance_hidden),
            nn.SiLU(),
            nn.Conv2d(
                instance_hidden,
                instance_hidden,
                3,
                padding=1,
                groups=instance_hidden,
                bias=False,
            ),
            nn.GroupNorm(num_groups, instance_hidden),
            nn.SiLU(),
            nn.Conv2d(instance_hidden, channels, 1, bias=False),
        )
        self.base_scale = nn.Parameter(torch.tensor(float(base_scale)))
        self.class_scale = nn.Parameter(torch.tensor(float(class_scale)))
        self.instance_scale = nn.Parameter(torch.tensor(float(instance_scale)))
        self.log_global_gain = nn.Parameter(torch.zeros(()))
        if target_rms is not None:
            self._calibrate_gain(float(target_rms))

    @property
    def global_gain(self) -> Tensor:
        return self.log_global_gain.exp()

    def components_from_probabilities(
        self, probabilities: Tensor, reference: Tensor
    ) -> dict[str, Tensor]:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(
                f"expected probabilities [B,{self.num_classes}], got {tuple(probabilities.shape)}"
            )
        expected = (self.channels, self.spatial_size, self.spatial_size)
        if reference.ndim != 4 or tuple(reference.shape[1:]) != expected:
            raise ValueError(
                f"expected reference [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(reference.shape)}"
            )
        batch = probabilities.shape[0]
        base = self.base_state.expand(batch, -1, -1, -1)
        centered = probabilities - 1.0 / self.num_classes
        latent = self.class_channel(centered @ self.class_embedding)
        spatial = latent[:, :, None, None].expand(
            -1, -1, self.spatial_size, self.spatial_size
        )
        class_component = self.class_spatial(spatial)
        instance = self.instance_encoder(reference.detach())
        weighted_base = self.base_scale * base
        weighted_class = self.class_scale * class_component
        weighted_instance = self.instance_scale * instance
        return {
            "base": base,
            "class": class_component,
            "instance": instance,
            "weighted_base": weighted_base,
            "weighted_class": weighted_class,
            "weighted_instance": weighted_instance,
            "combined": weighted_base + weighted_class + weighted_instance,
            "centered_probabilities": centered,
        }

    @torch.no_grad()
    def _calibrate_gain(self, target_rms: float) -> None:
        if not math.isfinite(target_rms) or target_rms <= 0:
            raise ValueError("target_rms must be finite and positive")
        logits = torch.zeros(1, self.num_classes)
        logits[0, 0] = 4.0
        generator = torch.Generator().manual_seed(1729)
        reference = 0.6 * torch.randn(
            1,
            self.channels,
            self.spatial_size,
            self.spatial_size,
            generator=generator,
        )
        combined = self.components_from_probabilities(
            logits.softmax(dim=-1), reference
        )["combined"]
        current = combined.square().mean().sqrt().clamp_min(1e-12)
        self.log_global_gain.fill_(math.log(target_rms / float(current)))

    def forward(
        self,
        logits: Tensor,
        reference: Tensor,
        *,
        return_components: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        probabilities = logits.softmax(dim=-1)
        components = self.components_from_probabilities(probabilities, reference)
        boundary = self.global_gain * components["combined"]
        if not return_components:
            return boundary, probabilities
        details = dict(components)
        details["boundary"] = boundary
        details["global_gain"] = self.global_gain
        return boundary, probabilities, details

class ResNet50Counterflow(nn.Module):
    """ResNet-50 whose third stage is a persistent two-boundary lattice."""

    def __init__(
        self,
        *,
        topology: Topology = "countercurrent",
        num_classes: int = 10,
        lattice_blocks: int = 5,
        num_groups: int = 32,
        beta: float = 0.1,
        inverse_steps: int = 15,
        train_inner_steps: int = 8,
        eval_max_steps: int = 8,
        num_outer_train: int = 1,
        num_outer_eval: int = 1,
        tolerance: float = 1e-3,
        damping: float = 0.5,
        conductance_max: float = 0.49,
        conductance_init_logit: float = -2.2,
        exchange_enabled: bool = True,
        class_rank: int = 32,
        boundary_base_scale: float = 0.15,
        boundary_class_scale: float = 1.0,
        boundary_instance_scale: float = 0.1,
        boundary_target_rms: float | None = 0.55,
    ) -> None:
        super().__init__()
        if topology not in ("countercurrent", "cocurrent"):
            raise ValueError(f"unsupported topology {topology!r}")
        if lattice_blocks <= 0:
            raise ValueError("lattice_blocks must be positive")
        if min(train_inner_steps, eval_max_steps, num_outer_train, num_outer_eval) <= 0:
            raise ValueError("step counts must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        self.topology: Topology = topology
        self.depth = int(lattice_blocks)
        self.train_inner_steps = int(train_inner_steps)
        self.eval_max_steps = int(eval_max_steps)
        self.num_outer_train = int(num_outer_train)
        self.num_outer_eval = int(num_outer_eval)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.exchange_enabled = bool(exchange_enabled)

        # ---- stock ResNet-50, CIFAR adaptation ------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_stage(64, 64, blocks=3, stride=1)
        self.layer2 = _make_stage(64 * EXPANSION, 128, blocks=4, stride=2)
        self.stage_entry = _stage_entry()

        # ---- counterflow lattice replaces layer3 blocks 1..N ----------------
        self.F = nn.ModuleList(
            InvertibleBottleneck(
                LATTICE_CHANNELS,
                BOTTLENECK_CHANNELS,
                num_groups=num_groups,
                beta=beta,
                inverse_steps=inverse_steps,
            )
            for _ in range(self.depth)
        )
        self.G = nn.ModuleList(ExactInverseBottleneck(block) for block in self.F)
        self.exchange = nn.ModuleList(
            ChannelwiseConductance(
                LATTICE_CHANNELS,
                conductance_max=conductance_max,
                initial_logit=conductance_init_logit,
            )
            for _ in range(self.depth)
        )

        # ---- stock ResNet-50 tail -------------------------------------------
        self.layer4 = _make_stage(LATTICE_CHANNELS, 512, blocks=3, stride=2)
        self.fc = nn.Linear(512 * EXPANSION, num_classes)
        self.hypothesis_head = nn.Linear(LATTICE_CHANNELS, num_classes)

        self.target_boundary = LightweightTargetBoundary(
            num_classes=num_classes,
            channels=LATTICE_CHANNELS,
            spatial_size=8,
            class_rank=class_rank,
            num_groups=num_groups,
            base_scale=boundary_base_scale,
            class_scale=boundary_class_scale,
            instance_scale=boundary_instance_scale,
            target_rms=boundary_target_rms,
        )

    # ------------------------------------------------------------------ pieces

    @property
    def boundary_index(self) -> int:
        return self.depth if self.topology == "countercurrent" else 0

    def _front(self, x: Tensor) -> Tensor:
        """Everything before the lattice: 32x32 image -> 8x8x1024 source inlet."""
        out = self.layer2(self.layer1(self.stem(x)))
        return self.stage_entry(out)

    def _classify(self, lattice_state: Tensor) -> Tensor:
        """Final prediction, using the same tail as plain ResNet-50."""
        out = self.layer4(lattice_state)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)

    def _hypothesis(self, lattice_state: Tensor) -> Tensor:
        """Cheap class hypothesis used to build the target boundary each step."""
        return self.hypothesis_head(lattice_state.mean(dim=(2, 3)))

    def _pure_forward(self, source: Tensor) -> list[Tensor]:
        states = [source]
        for block in self.F:
            states.append(block(states[-1]))
        return states

    def _initialize_c(self, boundary: Tensor) -> list[Tensor]:
        states: list[Tensor | None] = [None] * (self.depth + 1)
        if self.topology == "countercurrent":
            states[self.depth] = boundary
            for layer in reversed(range(self.depth)):
                inlet = states[layer + 1]
                if inlet is None:
                    raise RuntimeError("undefined reverse inlet")
                states[layer] = self.G[layer](inlet)
        else:
            states[0] = boundary
            for layer in range(self.depth):
                inlet = states[layer]
                if inlet is None:
                    raise RuntimeError("undefined co-current inlet")
                states[layer + 1] = self.G[layer](inlet)
        result = [state for state in states if state is not None]
        if len(result) != self.depth + 1:
            raise RuntimeError("C initialization left an undefined state")
        return result

    def _exchange_index(self, layer: int) -> int:
        """Which C state pairs with H[layer + 1] under this topology."""
        return layer if self.topology == "countercurrent" else layer + 1

    def _transport_c(self, c_old: list[Tensor]) -> list[Tensor]:
        """Synchronous reverse transport of the previous sweep's C field."""
        transported: list[Tensor | None] = [None] * (self.depth + 1)
        if self.topology == "countercurrent":
            transported[self.depth] = c_old[self.depth]
            for layer in reversed(range(self.depth)):
                inlet = transported[layer + 1]
                if inlet is None:
                    raise RuntimeError("undefined reverse inlet")
                transported[layer] = self.G[layer](inlet)
        else:
            transported[0] = c_old[0]
            for layer in range(self.depth):
                inlet = transported[layer]
                if inlet is None:
                    raise RuntimeError("undefined co-current inlet")
                transported[layer + 1] = self.G[layer](inlet)
        return [state for state in transported if state is not None]

    def _interior_pairs(
        self, h_states: list[Tensor], c_states: list[Tensor]
    ) -> list[Tensor]:
        c_interior = (
            c_states[: self.depth]
            if self.topology == "countercurrent"
            else c_states[1:]
        )
        return h_states[1:] + c_interior

    def _residual_per_sample(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        h_candidate: list[Tensor],
        c_candidate: list[Tensor],
    ) -> Tensor:
        current = self._interior_pairs(h_states, c_states)
        candidate = self._interior_pairs(h_candidate, c_candidate)
        numerator = sum(
            (new.float() - old.float()).flatten(1).square().sum(dim=1)
            for old, new in zip(current, candidate, strict=True)
        )
        denominator = sum(
            state.float().flatten(1).square().sum(dim=1) for state in current
        )
        return (numerator / (denominator + 1e-12)).sqrt()

    def _damped_update(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        h_candidate: list[Tensor],
        c_candidate: list[Tensor],
        *,
        source: Tensor,
        boundary: Tensor,
        update_mask: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        def update(old: Tensor, candidate: Tensor) -> Tensor:
            damped = old + self.damping * (candidate - old)
            if update_mask is None:
                return damped
            return torch.where(update_mask.reshape(-1, 1, 1, 1), damped, old)

        new_h = [source]
        new_h.extend(
            update(old, candidate)
            for old, candidate in zip(h_states[1:], h_candidate[1:], strict=True)
        )
        new_c = list(c_states)
        interior = (
            range(self.depth) if self.topology == "countercurrent" else range(1, self.depth + 1)
        )
        for index in interior:
            new_c[index] = update(c_states[index], c_candidate[index])
        new_c[self.boundary_index] = boundary
        step_residual = self._residual_per_sample(h_states, c_states, new_h, new_c)
        return new_h, new_c, step_residual

    def _sweep(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        *,
        reverse_off: bool,
    ) -> tuple[list[Tensor], list[Tensor], list[dict[str, Tensor]]]:
        """One synchronous Jacobi sweep of transport, exchange and damping."""
        h_bar = [h_states[0]]
        for layer in range(self.depth):
            h_bar.append(self.F[layer](h_states[layer]))
        c_bar = self._transport_c(c_states)

        h_candidate: list[Tensor] = [h_states[0]]
        c_candidate: list[Tensor] = list(c_bar)
        cells: list[dict[str, Tensor]] = []
        for layer in range(self.depth):
            index = self._exchange_index(layer)
            partner = c_bar[index]
            discrepancy = h_bar[layer + 1] - partner
            if self.exchange_enabled and not reverse_off:
                h_out, c_out, flux = self.exchange[layer].paired_update(
                    h_bar[layer + 1], partner
                )
            else:
                flux = torch.zeros_like(discrepancy)
                h_out, c_out = h_bar[layer + 1], partner
            h_candidate.append(h_out)
            c_candidate[index] = c_out
            cells.append(
                {
                    "h_bar": h_bar[layer + 1],
                    "c_bar": partner,
                    "D": discrepancy,
                    "Q": flux,
                }
            )
        return h_candidate, c_candidate, cells

    # ------------------------------------------------------------------ solver

    def solve(
        self,
        x: Tensor,
        *,
        steps: int,
        tolerance: float | None = None,
        reverse_off: bool = False,
        capture_diagnostics: bool = False,
        capture_states: bool = True,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """Run the two-boundary solve; return (logits, per-step logits, trace)."""
        source = self._front(x)
        h_states = self._pure_forward(source)
        logits = self._classify(h_states[-1])
        iteration_logits = [logits]

        boundary_result = self.target_boundary(
            self._hypothesis(h_states[-1]),
            h_states[-1],
            return_components=capture_diagnostics,
        )
        if capture_diagnostics:
            boundary, probabilities, boundary_components = boundary_result
        else:
            boundary, probabilities = boundary_result
            boundary_components = None
        c_states = self._initialize_c(boundary)

        history: list[dict[str, Any]] = []
        recorded: list[dict[str, Any]] = []
        if capture_diagnostics:
            recorded.append(
                self._record(
                    h_states,
                    c_states,
                    [],
                    boundary,
                    probabilities,
                    boundary_components,
                    include_states=capture_states,
                )
            )

        active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        steps_per_sample = torch.zeros(
            x.shape[0], dtype=torch.int64, device=x.device
        )

        for step in range(1, steps + 1):
            h_candidate, c_candidate, cells = self._sweep(
                h_states, c_states, reverse_off=reverse_off
            )
            h_states, c_states, step_residual = self._damped_update(
                h_states,
                c_states,
                h_candidate,
                c_candidate,
                source=source,
                boundary=boundary,
                update_mask=active if tolerance is not None else None,
            )
            logits = self._classify(h_states[-1])
            iteration_logits.append(logits)

            if tolerance is not None:
                steps_per_sample += active.to(steps_per_sample.dtype)
                converged_now = active & step_residual.lt(tolerance)
                active = active & ~converged_now
                converged = ~active
            else:
                converged_now = torch.zeros_like(active)
                converged = torch.zeros_like(active)
                steps_per_sample += 1
            if tolerance is not None:
                done = not bool(active.any())
            else:
                done = False
            history.append(
                {
                    "step": step,
                    "residual": step_residual.detach(),
                    "active": active.detach(),
                    "converged_now": converged_now.detach(),
                    "cells": cells,
                    "converged": converged.detach(),
                }
            )
            if capture_diagnostics:
                recorded.append(
                    self._record(
                        h_states,
                        c_states,
                        cells,
                        boundary,
                        probabilities,
                        boundary_components,
                        include_states=capture_states,
                    )
                )
            if done:
                break

        trace: dict[str, Any] = {
            "history": history,
            "steps": len(history),
            "steps_per_sample": steps_per_sample.detach().cpu(),
            "converged": converged.detach().cpu() if history else torch.zeros(x.shape[0], dtype=torch.bool),
            "boundary_components": boundary_components,
        }
        if capture_diagnostics:
            trace["recorded"] = recorded
        return logits, iteration_logits, trace

    def _record(
        self,
        h_states: list[Tensor],
        c_states: list[Tensor],
        cells: list[dict[str, Tensor]],
        boundary: Tensor | None,
        probabilities: Tensor | None,
        boundary_components: dict[str, Tensor] | None,
        *,
        include_states: bool,
    ) -> dict[str, Any]:
        def rms(state: Tensor) -> Tensor:
            return state.detach().float().flatten(1).square().mean(dim=1).sqrt().cpu()

        def cosine(left: Tensor, right: Tensor) -> Tensor:
            return F.cosine_similarity(
                left.detach().float().flatten(1),
                right.detach().float().flatten(1),
                dim=1,
            ).cpu()

        record: dict[str, Any] = {
            "H_norm": torch.stack([rms(state) for state in h_states], dim=1),
            "C_norm": torch.stack([rms(state) for state in c_states], dim=1),
            "cells": [
                {
                    "q_norm": rms(cell["Q"]),
                    "discrepancy": rms(cell["D"]),
                    "cos_d_u_h": cosine(cell["D"], cell["h_bar"]),
                }
                for cell in cells
            ],
            "boundary": None if boundary is None else boundary.detach().cpu(),
            "probabilities": (
                None if probabilities is None else probabilities.detach().cpu()
            ),
            "boundary_components": boundary_components,
        }
        if include_states:
            record["H"] = [state.detach().cpu() for state in h_states]
            record["C"] = [state.detach().cpu() for state in c_states]
        return record

    def run_solver(self, x: Tensor) -> tuple[Tensor, list[dict[str, Any]]]:
        """Public hook for the R1-R4 diagnostic: residual history over max steps."""
        logits, _, trace = self.solve(x, steps=self.eval_max_steps)
        return logits, [
            {
                "step": entry["step"],
                "relative_state_change": float(entry["residual"].mean()),
                "D": [cell["D"] for cell in entry["cells"]],
            }
            for entry in trace["history"]
        ]

    def _diagnostics(
        self,
        iteration_logits: list[Tensor],
        trace: dict[str, Any],
    ) -> dict[str, Tensor]:
        history = trace["history"]
        recorded = trace.get("recorded", [])
        diagnostics: dict[str, Tensor] = {
            "iter_logits": torch.stack(
                [logits.detach().cpu() for logits in iteration_logits]
            ),
            "initial_logits": iteration_logits[0].detach().cpu(),
            "solve_steps_per_sample": trace["steps_per_sample"],
            "converged": trace["converged"],
            "gamma": torch.stack(
                [cell.gamma.detach().cpu().flatten() for cell in self.exchange]
            ),
        }
        if recorded:
            diagnostics["H_norm"] = torch.stack(
                [record["H_norm"] for record in recorded]
            ).mean(dim=1)
            diagnostics["C_norm"] = torch.stack(
                [record["C_norm"] for record in recorded]
            ).mean(dim=1)
        if history:
            residual = torch.stack(
                [entry["residual"].detach().cpu() for entry in history]
            )
            diagnostics["equation_residual"] = residual
            diagnostics["step_residual"] = residual
            diagnostics["residual"] = residual.mean(dim=1)
            for key in ("q_norm", "discrepancy", "cos_d_u_h"):
                values = torch.stack(
                    [
                        torch.stack([cell[key] for cell in record["cells"]], dim=1)
                        for record in recorded[1:]
                    ]
                )
                diagnostics[key] = values.mean(dim=1)

        components = trace.get("boundary_components")
        if isinstance(components, dict):
            boundary_norm = components["boundary"].detach().float().flatten(1).norm(dim=1)
            gain = components["global_gain"].detach().float()
            raw = torch.stack(
                [
                    components[name].detach().float().flatten(1).square().mean(dim=1).sqrt()
                    for name in ("base", "class", "instance")
                ],
                dim=1,
            )
            weighted = torch.stack(
                [
                    (gain * components[name].detach().float()).flatten(1).norm(dim=1)
                    for name in ("weighted_base", "weighted_class", "weighted_instance")
                ],
                dim=1,
            )
            diagnostics["boundary_component_rms"] = raw.mean(dim=0, keepdim=True).cpu()
            diagnostics["boundary_component_norm"] = weighted.mean(dim=0, keepdim=True).cpu()
            diagnostics["boundary_component_ratio"] = (
                weighted / boundary_norm[:, None].clamp_min(1e-12)
            ).mean(dim=0, keepdim=True).cpu()
            diagnostics["boundary_global_gain"] = gain.reshape(1).cpu()
        return diagnostics

    # ----------------------------------------------------------------- forward

    def _run(
        self,
        x: Tensor,
        *,
        capture_diagnostics: bool,
        reverse_off: bool,
        adaptive: bool,
        fixed_steps: int | None = None,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        if not self.exchange_enabled:
            # Architecture control.  With the exchange removed the reverse
            # field can never influence the forward state, so running the
            # solver would only repeat an identical forward pass.  Report a
            # single pass through the invertible stage instead, which is the
            # architecture this control is meant to isolate.
            source = self._front(x)
            states = self._pure_forward(source)
            logits = self._classify(states[-1])
            trace: dict[str, Any] = {"history": [], "steps": 0}
            if capture_diagnostics:
                trace["recorded"] = [
                    self._record(
                        states,
                        states,
                        [],
                        source,
                        None,
                        None,
                        include_states=False,
                    )
                ]
                trace["steps_per_sample"] = torch.zeros(
                    x.shape[0], dtype=torch.int64
                )
                trace["converged"] = torch.ones(x.shape[0], dtype=torch.bool)
            return logits, [logits], trace
        if fixed_steps is not None:
            if fixed_steps < 0:
                raise ValueError("fixed_steps must be nonnegative")
            return self.solve(
                x,
                steps=fixed_steps,
                tolerance=None,
                reverse_off=reverse_off,
                capture_diagnostics=capture_diagnostics,
                capture_states=False,
            )
        if not adaptive:
            steps = self.train_inner_steps * self.num_outer_train
            return self.solve(
                x,
                steps=steps,
                tolerance=None,
                reverse_off=reverse_off,
                capture_diagnostics=capture_diagnostics,
                capture_states=False,
            )
        return self.solve(
            x,
            steps=self.eval_max_steps * self.num_outer_eval,
            tolerance=self.tolerance,
            reverse_off=reverse_off,
            capture_diagnostics=capture_diagnostics,
            capture_states=False,
        )

    def predict_iterations(
        self,
        x: Tensor,
        *,
        reverse_off: bool = False,
        steps: int | None = None,
    ) -> Tensor:
        _, logits, _ = self._run(
            x,
            capture_diagnostics=False,
            reverse_off=reverse_off,
            adaptive=False,
            fixed_steps=steps,
        )
        return torch.stack(logits)

    def forward(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool = False,
        reverse_off: bool = False,
        return_initial_logits: bool = False,
        adaptive: bool | None = None,
        fixed_steps: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Any]]:
        if return_diagnostics and return_initial_logits:
            raise ValueError("request diagnostics or initial logits, not both")
        if adaptive is None:
            adaptive = not self.training
        logits, iterations, diagnostics = self._run(
            x,
            capture_diagnostics=return_diagnostics,
            reverse_off=reverse_off,
            adaptive=adaptive,
            fixed_steps=fixed_steps,
        )
        if return_initial_logits:
            return logits, iterations[0]
        if return_diagnostics:
            return logits, self._diagnostics(iterations, diagnostics)
        return logits

    def extra_repr(self) -> str:
        return (
            f"topology={self.topology}, lattice_blocks={self.depth}, "
            f"exchange_enabled={self.exchange_enabled}, "
            f"train_inner_steps={self.train_inner_steps}, damping={self.damping}"
        )


class ResNet50Countercurrent(ResNet50Counterflow):
    """Counterflow: C enters at the target end and is transported L -> 0."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["topology"] = "countercurrent"
        super().__init__(**kwargs)


class ResNet50Cocurrent(ResNet50Counterflow):
    """Same solver, boundary and exchange law; C is transported 0 -> L."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["topology"] = "cocurrent"
        super().__init__(**kwargs)


class ResNet50LatticeControl(ResNet50Counterflow):
    """Architecture control: identical invertible stage, exchange switched off."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["topology"] = "countercurrent"
        kwargs["exchange_enabled"] = False
        super().__init__(**kwargs)


class ResNet50Plain(nn.Module):
    """Stock ResNet-50 with the CIFAR adaptation; the baseline every arm is compared to."""

    def __init__(self, *, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_stage(64, 64, blocks=3, stride=1)
        self.layer2 = _make_stage(64 * EXPANSION, 128, blocks=4, stride=2)
        self.layer3 = _make_stage(128 * EXPANSION, 256, blocks=6, stride=2)
        self.layer4 = _make_stage(256 * EXPANSION, 512, blocks=3, stride=2)
        self.fc = nn.Linear(512 * EXPANSION, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer2(self.layer1(self.stem(x)))
        out = self.layer4(self.layer3(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


__all__ = [
    "ExactInverseBottleneck",
    "InvertibleBottleneck",
    "ResNet50Cocurrent",
    "ResNet50Countercurrent",
    "ResNet50Counterflow",
    "ResNet50LatticeControl",
    "ResNet50Plain",
    "Topology",
]
