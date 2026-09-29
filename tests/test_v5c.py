"""Behavioral tests for V5-C persistent canonical counterflow."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from countercurrent_nn.config import load_config
from countercurrent_nn.engine import build_optimizer
from countercurrent_nn.models import build_model
from countercurrent_nn.models.canonical_exchange import (
    CanonicalExchange,
    OrthogonalChannelTransform,
)
from countercurrent_nn.models.decomposed_boundary import DecomposedTargetBoundary
from countercurrent_nn.profiling import count_macs
from countercurrent_nn.utils import parameter_counts


torch.set_num_threads(1)


def tiny(name: str = "persistent_canonical_countercurrent", **overrides):
    kwargs = dict(
        channels=8,
        depth=2,
        num_classes=4,
        num_groups=2,
        class_rank=4,
        lattice_size=8,
        train_inner_steps=3,
        eval_max_steps=5,
        num_outer_train=1,
        num_outer_eval=1,
        tolerance=1e-3,
        damping=0.5,
    )
    kwargs.update(overrides)
    return build_model(name, **kwargs)


class CanonicalExchangeTests(unittest.TestCase):
    def test_canonical_exchange_is_antisymmetric(self):
        module = CanonicalExchange(8)
        h_bar = torch.randn(2, 8, 4, 4)
        c_bar = torch.randn_like(h_bar)
        _, _, diagnostics = module(h_bar, c_bar)
        torch.testing.assert_close(
            diagnostics["u_h_new"] + diagnostics["u_c_new"],
            diagnostics["u_h"] + diagnostics["u_c"],
        )

    def test_orthogonal_transform_preserves_norm_and_inverts(self):
        transform = OrthogonalChannelTransform(8)
        identity = torch.eye(8)
        torch.testing.assert_close(
            transform.weight.T @ transform.weight, identity, atol=1e-6, rtol=1e-6
        )
        state = torch.randn(2, 8, 4, 4)
        encoded = transform.encode(state)
        torch.testing.assert_close(encoded.norm(), state.norm(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(transform.decode(encoded), state, atol=1e-5, rtol=1e-5)

    def test_both_streams_have_reciprocal_influence(self):
        module = CanonicalExchange(4)
        h_bar = torch.randn(1, 4, 3, 3, requires_grad=True)
        c_bar = torch.randn(1, 4, 3, 3, requires_grad=True)
        h_new, c_new, _ = module(h_bar, c_bar)
        c_to_h = torch.autograd.grad(h_new.square().sum(), c_bar, retain_graph=True)[0]
        h_to_c = torch.autograd.grad(c_new.square().sum(), h_bar)[0]
        self.assertGreater(float(c_to_h.abs().sum()), 0.0)
        self.assertGreater(float(h_to_c.abs().sum()), 0.0)

    def test_compute_matched_raw_exchange_remains_raw(self):
        module = CanonicalExchange(
            8, use_canonical=False, match_canonical_compute=True
        )
        h_bar = torch.randn(2, 8, 4, 4)
        c_bar = torch.randn_like(h_bar)
        h_new, c_new, diagnostics = module(h_bar, c_bar)
        discrepancy = h_bar - c_bar
        flux = module.conductance(discrepancy)
        torch.testing.assert_close(diagnostics["D"], discrepancy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_new, h_bar - flux, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(c_new, c_bar + flux, atol=1e-5, rtol=1e-5)


class DecomposedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.boundary = DecomposedTargetBoundary(
            num_classes=4,
            channels=8,
            spatial_size=8,
            class_rank=4,
            num_groups=2,
            target_rms=None,
        )

    def test_uniform_hypothesis_has_zero_class_component(self):
        probabilities = torch.full((2, 4), 0.25)
        reference = torch.randn(2, 8, 8, 8)
        components = self.boundary.components_from_probabilities(
            probabilities, reference
        )
        torch.testing.assert_close(
            components["class"], torch.zeros_like(components["class"])
        )

    def test_class_component_is_spatial(self):
        probabilities = torch.tensor([[0.8, 0.1, 0.05, 0.05]])
        reference = torch.zeros(1, 8, 8, 8)
        component = self.boundary.components_from_probabilities(
            probabilities, reference
        )["class"]
        spatial_variance = component.flatten(2).var(dim=-1).sum()
        self.assertGreater(float(spatial_variance), 0.0)

    def test_instance_path_stops_gradient_to_reference(self):
        logits = torch.randn(2, 4, requires_grad=True)
        reference = torch.randn(2, 8, 8, 8, requires_grad=True)
        output, _ = self.boundary(logits, reference)
        reference_gradient = torch.autograd.grad(
            output.sum(), reference, allow_unused=True, retain_graph=True
        )[0]
        logits_gradient = torch.autograd.grad(output.sum(), logits)[0]
        self.assertIsNone(reference_gradient)
        self.assertGreater(float(logits_gradient.abs().sum()), 0.0)

    def test_global_gain_is_positive(self):
        self.assertGreater(float(self.boundary.global_gain), 0.0)

    def test_gain_calibration_matches_the_reference_target_rms(self):
        boundary = DecomposedTargetBoundary(
            num_classes=4,
            channels=8,
            spatial_size=8,
            class_rank=4,
            num_groups=2,
            target_rms=0.55,
        )
        logits = torch.zeros(1, 4)
        logits[0, 0] = 4.0
        generator = torch.Generator().manual_seed(1729)
        reference = 0.6 * torch.randn(1, 8, 8, 8, generator=generator)
        output, _ = boundary(logits, reference)
        self.assertAlmostEqual(float(output.square().mean().sqrt()), 0.55, places=5)


class PersistentSolverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.x = torch.randn(2, 3, 32, 32)

    def test_c_state_persists_into_the_next_reverse_transport(self):
        model = tiny().eval()
        source = model.stem(self.x)
        h_states = model._pure_forward(source)
        boundary, _, _ = model._build_boundary(
            model._classify(h_states[-1]), h_states[-1]
        )
        c_states = model._initialize_c(boundary)
        perturbed = list(c_states)
        perturbed[1] = perturbed[1] + 0.5
        _, original_c, _ = model._fixed_point_map(
            h_states, c_states, source=source, boundary=boundary
        )
        _, changed_c, _ = model._fixed_point_map(
            h_states, perturbed, source=source, boundary=boundary
        )
        self.assertFalse(torch.allclose(original_c[0], changed_c[0]))

    def test_boundary_is_called_once_for_one_outer_solve(self):
        model = tiny(tolerance=1e9).eval()
        calls = 0

        def count(*_args):
            nonlocal calls
            calls += 1

        handle = model.target_boundary.register_forward_hook(count)
        try:
            model(self.x)
        finally:
            handle.remove()
        self.assertEqual(calls, 1)

    def test_both_boundaries_are_clamped_at_every_recorded_step(self):
        model = tiny().eval()
        _, diagnostics = model(self.x, return_diagnostics=True)
        source = model.stem(self.x)
        for recorded in diagnostics["H_history"][:, 0]:
            torch.testing.assert_close(recorded, source)
        boundary = diagnostics["boundary_history"][0]
        for recorded in diagnostics["C_history"][:, model.boundary_index]:
            torch.testing.assert_close(recorded, boundary)

    def test_training_uses_fixed_steps_even_when_tolerance_is_huge(self):
        model = tiny(tolerance=1e9).train()
        _, iterations, trace = model.unroll_train(self.x)
        self.assertEqual(len(trace["history"]), model.train_inner_steps)
        self.assertEqual(len(iterations), model.train_inner_steps + 1)
        self.assertTrue(
            torch.equal(
                trace["steps_per_sample"],
                torch.full((2,), model.train_inner_steps, dtype=torch.int64),
            )
        )

    def test_evaluation_can_stop_early(self):
        model = tiny(tolerance=1e9).eval()
        _, iterations, trace = model.solve_eval(self.x)
        self.assertEqual(len(trace["history"]), 1)
        self.assertEqual(len(iterations), 2)
        self.assertTrue(torch.equal(trace["steps_per_sample"], torch.ones(2, dtype=torch.int64)))
        self.assertTrue(bool(trace["converged"].all()))
        self.assertEqual(model.predict_iterations(self.x).shape[0], model.eval_max_steps + 1)

    def test_evaluation_stops_samples_independently(self):
        model = tiny(eval_max_steps=3).eval()
        equation_first = torch.tensor([0.0, 1.0])
        step_first = torch.tensor([0.0, 0.5])
        equation_second = torch.tensor([0.0, 0.0])
        step_second = torch.tensor([0.0, 0.0])
        with patch.object(
            model,
            "_residual_per_sample",
            side_effect=[equation_first, step_first, equation_second, step_second],
        ):
            _, _, trace = model.solve_eval(self.x)
        self.assertTrue(
            torch.equal(trace["steps_per_sample"], torch.tensor([1, 2]))
        )
        self.assertTrue(bool(trace["converged"].all()))

    def test_equation_and_step_residual_are_separate(self):
        model = tiny(damping=0.5).eval()
        _, _, trace = model.solve_eval(self.x)
        first = trace["history"][0]
        torch.testing.assert_close(
            first["step_residual"], 0.5 * first["equation_residual"],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_no_ground_truth_interface(self):
        model = tiny()
        forbidden = {"label", "labels", "target", "targets", "teacher", "y"}
        self.assertTrue(forbidden.isdisjoint(inspect.signature(model.forward).parameters))
        with self.assertRaises(TypeError):
            model(self.x, torch.zeros(2, dtype=torch.long))

    def test_gradients_reach_every_parameter(self):
        model = tiny(train_inner_steps=2).train()
        labels = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(model(self.x), labels)
        loss.backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ]
        self.assertEqual(missing, [])


class V5CFairnessTests(unittest.TestCase):
    def test_countercurrent_and_cocurrent_match_parameters_and_macs(self):
        counter = tiny("persistent_canonical_countercurrent").eval()
        co = tiny("persistent_canonical_cocurrent").eval()
        self.assertEqual(parameter_counts(counter), parameter_counts(co))
        self.assertEqual(
            {key: value.shape for key, value in counter.state_dict().items()},
            {key: value.shape for key, value in co.state_dict().items()},
        )
        sample = torch.zeros(1, 3, 32, 32)
        self.assertEqual(count_macs(counter, sample), count_macs(co, sample))

    def test_compute_matched_raw_control_matches_canonical_budget(self):
        root = Path(__file__).parents[1] / "countercurrent_nn" / "configs"
        canonical_config = load_config(root / "cifar10_v5c_countercurrent.yaml")
        raw_config = load_config(root / "cifar10_v5a_countercurrent_raw.yaml")
        canonical = build_model(canonical_config["model"]).eval()
        raw = build_model(raw_config["model"]).eval()
        self.assertEqual(parameter_counts(canonical), parameter_counts(raw))
        self.assertEqual(
            {key: value.shape for key, value in canonical.state_dict().items()},
            {key: value.shape for key, value in raw.state_dict().items()},
        )
        sample = torch.zeros(1, 3, 32, 32)
        self.assertEqual(count_macs(canonical, sample), count_macs(raw, sample))

    def test_boundary_mechanism_scalars_have_zero_weight_decay(self):
        model = tiny()
        optimizer = build_optimizer(
            model,
            {
                "optimizer": "sgd",
                "lr": 0.1,
                "weight_decay": 5e-4,
                "conductance_weight_decay": 0.0,
                "boundary_scalar_weight_decay": 0.0,
            },
        )
        decay_by_id = {
            id(parameter): float(group["weight_decay"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        boundary = model.target_boundary
        for parameter in (
            boundary.base_scale,
            boundary.class_scale,
            boundary.instance_scale,
            boundary.log_global_gain,
        ):
            self.assertEqual(decay_by_id[id(parameter)], 0.0)
        self.assertEqual(decay_by_id[id(model.exchange[0].conductance.logit_gamma)], 0.0)
        self.assertEqual(decay_by_id[id(model.head.weight)], 5e-4)

    def test_primary_configs_differ_only_in_identity(self):
        root = Path(__file__).parents[1] / "countercurrent_nn" / "configs"
        with (root / "cifar10_v5c_countercurrent.yaml").open(encoding="utf-8") as handle:
            counter = yaml.safe_load(handle)
        with (root / "cifar10_v5c_cocurrent.yaml").open(encoding="utf-8") as handle:
            co = yaml.safe_load(handle)
        self.assertEqual(counter["dataset"], co["dataset"])
        self.assertEqual(counter["training"], co["training"])
        counter_model = dict(counter["model"])
        co_model = dict(co["model"])
        counter_model.pop("name")
        co_model.pop("name")
        self.assertEqual(counter_model, co_model)


if __name__ == "__main__":
    unittest.main()
