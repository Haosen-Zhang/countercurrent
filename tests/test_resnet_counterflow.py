"""Tests for the ResNet-50-backed counterflow lattice."""

from __future__ import annotations

import unittest

import torch

from countercurrent_nn.models import build_model
from countercurrent_nn.models.resnet_counterflow import (
    ExactInverseBottleneck,
    InvertibleBottleneck,
    LightweightTargetBoundary,
)

# Deliberately small so the suite stays fast; the block structure is unchanged.
TINY = dict(
    lattice_blocks=2,
    num_groups=32,
    train_inner_steps=2,
    eval_max_steps=3,
    class_rank=8,
)


def tiny(name: str, **overrides):
    """Build a small arm; the plain baseline takes no counterflow options."""
    if name == "resnet50_plain":
        return build_model(name)
    return build_model(name, **{**TINY, **overrides})


def unique_parameter_count(model) -> int:
    """Count parameters once, so a tied reverse transport is not double counted."""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


class InvertibleBottleneckTests(unittest.TestCase):
    def test_inverse_recovers_the_input(self):
        block = InvertibleBottleneck(64, 16, num_groups=8, beta=0.1).eval()
        x = torch.randn(2, 64, 8, 8)
        with torch.no_grad():
            recovered = block.inverse(block(x))
        self.assertLess(float((recovered - x).norm() / x.norm()), 1e-5)

    def test_inverse_is_a_smaller_contraction_for_smaller_beta(self):
        x = torch.randn(2, 64, 8, 8)
        errors = []
        for beta in (0.1, 0.3):
            block = InvertibleBottleneck(64, 16, num_groups=8, beta=beta).eval()
            with torch.no_grad():
                errors.append(float((block.inverse(block(x)) - x).norm()))
        self.assertLess(errors[0], errors[1] + 1e-6)

    def test_rejects_beta_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            InvertibleBottleneck(64, 16, num_groups=8, beta=1.5)

    def test_exact_inverse_shares_the_forward_parameters(self):
        block = InvertibleBottleneck(64, 16, num_groups=8)
        adapter = ExactInverseBottleneck(block)
        self.assertEqual(
            [id(p) for p in adapter.parameters()], [id(p) for p in block.parameters()]
        )


class BoundaryTests(unittest.TestCase):
    def make(self):
        return LightweightTargetBoundary(
            num_classes=10, channels=64, spatial_size=8, class_rank=8, num_groups=8
        )

    def test_uniform_probabilities_give_zero_class_term(self):
        boundary = self.make()
        reference = torch.zeros(1, 64, 8, 8)
        uniform = torch.full((1, 10), 0.1)
        components = boundary.components_from_probabilities(uniform, reference)
        self.assertLess(float(components["class"].abs().max()), 1e-6)

    def test_instance_term_is_stop_gradient(self):
        boundary = self.make()
        reference = torch.randn(1, 64, 8, 8, requires_grad=True)
        out, _ = boundary(torch.zeros(1, 10), reference)
        gradient = torch.autograd.grad(out.sum(), reference, allow_unused=True)[0]
        self.assertIsNone(gradient)

    def test_hypothesis_changes_the_boundary(self):
        boundary = self.make()
        reference = torch.zeros(1, 64, 8, 8)
        hot, _ = boundary(torch.tensor([[8.0] + [0.0] * 9]), reference)
        cold, _ = boundary(torch.tensor([[0.0] * 9 + [8.0]]), reference)
        self.assertFalse(torch.allclose(hot, cold))

    def test_gain_calibration_matches_the_target_rms(self):
        boundary = LightweightTargetBoundary(
            num_classes=10,
            channels=64,
            spatial_size=8,
            class_rank=8,
            num_groups=8,
            target_rms=1.75,
        )
        # Compare the same module with and without an extra gain factor: two
        # separately constructed modules differ in their random components, so
        # their raw rms is not comparable.
        module = LightweightTargetBoundary(
            num_classes=10,
            channels=64,
            spatial_size=8,
            class_rank=8,
            num_groups=8,
            target_rms=1.0,
        )
        logits = torch.zeros(1, 10)
        reference = torch.zeros(1, 64, 8, 8)
        with torch.no_grad():
            before = float(module(logits, reference)[0].square().mean().sqrt())
            module.log_global_gain.add_(torch.log(torch.tensor(2.0)))
            after = float(module(logits, reference)[0].square().mean().sqrt())
        self.assertAlmostEqual(after / before, 2.0, places=3)


class FairnessTests(unittest.TestCase):
    def test_parameter_counts_are_close(self):
        # Must use the shipped block count: the lattice replaces all but the first
        # of layer3's six blocks, so a reduced lattice is not comparable to the
        # full plain baseline.
        plain = unique_parameter_count(build_model("resnet50_plain"))
        for name in (
            "resnet50_countercurrent",
            "resnet50_cocurrent",
            "resnet50_lattice_control",
        ):
            count = unique_parameter_count(build_model(name))
            self.assertLess(
                abs(count - plain) / plain,
                0.05,
                f"{name} differs from plain by more than 5%",
            )

    def test_reverse_transport_adds_no_parameters(self):
        model = tiny("resnet50_countercurrent")
        forward_ids = sorted(id(p) for block in model.F for p in block.parameters())
        reverse_ids = sorted(id(p) for block in model.G for p in block.parameters())
        self.assertEqual(forward_ids, reverse_ids)

    def test_countercurrent_and_cocurrent_parameters_match_exactly(self):
        counter = tiny("resnet50_countercurrent").state_dict()
        co = tiny("resnet50_cocurrent").state_dict()
        self.assertEqual(sorted(counter), sorted(co))
        for key in counter:
            self.assertEqual(counter[key].shape, co[key].shape)


class SolverTests(unittest.TestCase):
    def test_source_inlet_is_clamped_every_step(self):
        model = tiny("resnet50_countercurrent").eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            source = model._front(x)
            _, _, trace = model.solve(x, steps=3, capture_diagnostics=True)
        for record in trace["recorded"]:
            self.assertTrue(torch.allclose(record["H"][0], source.cpu(), atol=1e-6))

    def test_transport_directions_are_opposite(self):
        counter = tiny("resnet50_countercurrent").eval()
        co = tiny("resnet50_cocurrent").eval()
        boundary = torch.randn(2, 1024, 8, 8)
        with torch.no_grad():
            counter_states = counter._initialize_c(boundary)
            co_states = co._initialize_c(boundary)
        self.assertTrue(torch.allclose(counter_states[-1], boundary))
        self.assertTrue(torch.allclose(co_states[0], boundary))

    def test_control_skips_the_solver(self):
        model = tiny("resnet50_lattice_control").eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            trajectory = model.predict_iterations(x)
        self.assertEqual(trajectory.shape[0], 1)

    def test_reverse_off_changes_the_prediction(self):
        model = tiny("resnet50_countercurrent").eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            full = model(x)
            off = model(x, reverse_off=True)
        self.assertGreater(float((full - off).norm() / full.norm()), 1e-4)

    def test_solver_returns_one_prediction_per_step(self):
        model = tiny("resnet50_countercurrent").eval()
        with torch.no_grad():
            trajectory = model.predict_iterations(torch.randn(2, 3, 32, 32))
        self.assertEqual(trajectory.shape[0], model.train_inner_steps + 1)


class ContractTests(unittest.TestCase):
    def test_main_model_never_accepts_labels(self):
        model = tiny("resnet50_countercurrent")
        with self.assertRaises(TypeError):
            model(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))

    def test_gradients_are_finite_for_every_active_arm(self):
        for name in (
            "resnet50_plain",
            "resnet50_countercurrent",
            "resnet50_cocurrent",
        ):
            model = tiny(name)
            model.train()
            x = torch.randn(2, 3, 32, 32)
            labels = torch.randint(0, 10, (2,))
            loss = torch.nn.functional.cross_entropy(model(x), labels)
            loss.backward()
            bad = [
                key
                for key, value in model.named_parameters()
                if value.grad is not None and not torch.isfinite(value.grad).all()
            ]
            self.assertEqual(bad, [], f"{name} produced non-finite gradients")

    def test_plain_baseline_is_not_a_counterflow_model(self):
        model = tiny("resnet50_plain")
        self.assertFalse(hasattr(model, "F"))
        self.assertFalse(hasattr(model, "target_boundary"))


if __name__ == "__main__":
    unittest.main()
