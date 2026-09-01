from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
import yaml

from countercurrent_nn.models import (
    ChannelwiseConductance,
    ClassicCNN,
    CocurrentCNN,
    CountercurrentCNN,
    ForwardOnlyRecurrentCNN,
    LearnedBoundaryCountercurrentCNN,
    NullBoundaryCountercurrentCNN,
    SelfGeneratedTargetBoundary,
    SingleStreamFeedForwardCNN,
    Stem,
)
from countercurrent_nn.profiling import count_macs
from countercurrent_nn.utils import parameter_counts


torch.set_num_threads(1)


def small_kwargs() -> dict[str, int | float | str]:
    return {
        "channels": 8,
        "depth": 3,
        "refine_steps": 2,
        "num_classes": 10,
        "num_groups": 2,
        "residual_scale": 0.1,
        "boundary_type": "self_generated",
        "prototype_dim": 8,
        "lattice_size": 8,
        "conductance_max": 0.49,
        "conductance_init_logit": -2.2,
    }


class ComponentTests(unittest.TestCase):
    def test_stem_has_documented_shape(self) -> None:
        output = Stem()(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 64, 8, 8))

    def test_soft_hypothesis_boundary_matches_manual_computation(self) -> None:
        module = SelfGeneratedTargetBoundary(
            num_classes=3, channels=4, prototype_dim=2
        )
        logits = torch.tensor([[1.0, 0.0, -1.0]])
        reference = torch.zeros(1, 4, 5, 5)
        boundary, probabilities = module(logits, reference)
        hypothesis = probabilities @ module.class_prototypes
        expected = module.projector(hypothesis)[:, :, None, None].expand_as(boundary)
        torch.testing.assert_close(boundary, expected)
        self.assertEqual(boundary.shape, reference.shape)
        self.assertFalse(hasattr(module, "argmax"))

    def test_soft_boundary_is_differentiable_with_respect_to_logits(self) -> None:
        module = SelfGeneratedTargetBoundary(
            num_classes=3, channels=4, prototype_dim=2
        )
        logits = torch.randn(2, 3, requires_grad=True)
        boundary, _ = module(logits, torch.zeros(2, 4, 2, 2))
        gradient = torch.autograd.grad(boundary.square().sum(), logits)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_conductance_range_and_initial_value(self) -> None:
        block = ChannelwiseConductance(channels=8)
        gamma = block.gamma
        self.assertGreater(float(gamma.min()), 0.0)
        self.assertLess(float(gamma.max()), 0.5)
        self.assertAlmostEqual(float(gamma.mean()), 0.49 * torch.sigmoid(torch.tensor(-2.2)).item(), places=6)
        self.assertEqual(sum(parameter.numel() for parameter in block.parameters()), 8)

    def test_paired_exchange_is_algebraically_zero_sum(self) -> None:
        block = ChannelwiseConductance(channels=8)
        h_bar = torch.randn(2, 8, 8, 8)
        c_bar = torch.randn_like(h_bar)
        h_out, c_out, _ = block.paired_update(h_bar, c_bar)
        torch.testing.assert_close(h_out + c_out, h_bar + c_bar)

    def test_exchange_has_reciprocal_local_influence(self) -> None:
        block = ChannelwiseConductance(channels=4)
        h_bar = torch.randn(1, 4, 3, 3, requires_grad=True)
        c_bar = torch.randn(1, 4, 3, 3, requires_grad=True)
        h_out, c_out, _ = block.paired_update(h_bar, c_bar)
        c_to_h = torch.autograd.grad(h_out.square().sum(), c_bar, retain_graph=True)[0]
        h_to_c = torch.autograd.grad(c_out.square().sum(), h_bar)[0]
        self.assertGreater(float(c_to_h.abs().sum()), 0.0)
        self.assertGreater(float(h_to_c.abs().sum()), 0.0)


class TwoSweepInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.x = torch.randn(2, 3, 32, 32)

    def test_diagnostic_shapes_separate_t0_from_refinement(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        logits, diagnostics = model(self.x, return_diagnostics=True)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(diagnostics["H_history"].shape, (3, 4, 2, 8, 8, 8))
        self.assertEqual(diagnostics["C_history"].shape, (2, 4, 2, 8, 8, 8))
        self.assertEqual(diagnostics["q_norm"].shape, (2, 3))
        self.assertEqual(diagnostics["discrepancy"].shape, (2, 3))
        self.assertEqual(diagnostics["exchange_energy"].shape, (2, 3))
        self.assertEqual(diagnostics["iter_logits"].shape, (3, 2, 10))

    def test_iteration_zero_is_a_pure_forward_pass(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        _, diagnostics = model(self.x, return_diagnostics=True)
        source = model.stem(self.x)
        expected_states = model._pure_forward(source)
        torch.testing.assert_close(
            diagnostics["H_history"][0],
            torch.stack(expected_states),
        )
        torch.testing.assert_close(
            diagnostics["initial_logits"],
            model._classify(expected_states[-1]),
        )

    def test_source_is_recomputed_and_clamped_each_iteration(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        stem_calls = 0

        def count_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
            nonlocal stem_calls
            stem_calls += 1

        handle = model.stem.register_forward_hook(count_call)
        try:
            _, diagnostics = model(self.x, return_diagnostics=True)
        finally:
            handle.remove()
        self.assertEqual(stem_calls, model.refine_steps + 1)
        expected = model.stem(self.x).detach()
        for state in diagnostics["H_history"][:, 0]:
            torch.testing.assert_close(state, expected)

    def test_boundary_is_generated_from_previous_iteration_logits(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        _, diagnostics = model(self.x, return_diagnostics=True)
        reference = diagnostics["H_history"][0, 0]
        for iteration in range(model.refine_steps):
            expected, probabilities = model.target_boundary(
                diagnostics["iter_logits"][iteration], reference
            )
            torch.testing.assert_close(
                diagnostics["boundary_history"][iteration], expected
            )
            torch.testing.assert_close(
                diagnostics["boundary_probabilities"][iteration], probabilities
            )

    def test_countercurrent_and_cocurrent_clamp_opposite_boundary_indices(self) -> None:
        for model_type, boundary_index in ((CountercurrentCNN, 3), (CocurrentCNN, 0)):
            model = model_type(**small_kwargs()).eval()
            _, diagnostics = model(self.x, return_diagnostics=True)
            for iteration in range(model.refine_steps):
                torch.testing.assert_close(
                    diagnostics["C_history"][iteration, boundary_index],
                    diagnostics["boundary_history"][iteration],
                )

    def test_reverse_off_reduces_every_refinement_to_pure_forward(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        _, diagnostics = model(
            self.x, return_diagnostics=True, reverse_off=True
        )
        for logits in diagnostics["iter_logits"][1:]:
            torch.testing.assert_close(logits, diagnostics["initial_logits"])
        torch.testing.assert_close(
            diagnostics["q_norm"], torch.zeros_like(diagnostics["q_norm"])
        )

    def test_reverse_path_changes_the_untrained_model_output(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        full = model(self.x)
        reverse_off = model(self.x, reverse_off=True)
        self.assertGreater(float((full - reverse_off).abs().sum()), 0.0)

    def test_final_head_reads_only_forward_outlet(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        captured: list[torch.Tensor] = []
        handle = model.head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().cpu())
        )
        try:
            _, diagnostics = model(self.x, return_diagnostics=True)
        finally:
            handle.remove()
        expected = diagnostics["H_history"][-1, -1].mean(dim=(2, 3))
        torch.testing.assert_close(captured[-1], expected)

    def test_main_model_has_no_label_or_teacher_input(self) -> None:
        parameters = inspect.signature(CountercurrentCNN.forward).parameters
        forbidden = {"label", "labels", "target", "targets", "teacher", "y"}
        self.assertTrue(forbidden.isdisjoint(parameters))

    def test_null_and_learned_boundary_ablations(self) -> None:
        null_model = NullBoundaryCountercurrentCNN(**small_kwargs()).eval()
        _, null_diagnostics = null_model(self.x, return_diagnostics=True)
        torch.testing.assert_close(
            null_diagnostics["boundary_history"],
            torch.zeros_like(null_diagnostics["boundary_history"]),
        )
        learned_model = LearnedBoundaryCountercurrentCNN(**small_kwargs()).eval()
        _, learned_diagnostics = learned_model(self.x, return_diagnostics=True)
        boundary = learned_diagnostics["boundary_history"]
        torch.testing.assert_close(boundary[:, 0], boundary[:, 1])

    def test_forward_only_control_has_t_plus_one_predictions(self) -> None:
        kwargs = small_kwargs()
        for key in (
            "boundary_type",
            "prototype_dim",
            "lattice_size",
            "conductance_max",
            "conductance_init_logit",
        ):
            kwargs.pop(key)
        model = ForwardOnlyRecurrentCNN(**kwargs).eval()
        _, diagnostics = model(self.x, return_diagnostics=True)
        self.assertEqual(diagnostics["iter_logits"].shape, (3, 2, 10))
        expected = model.stem(self.x)
        for state in diagnostics["H_history"][:, 0]:
            torch.testing.assert_close(state, expected)


class FairnessTests(unittest.TestCase):
    def test_countercurrent_and_cocurrent_have_identical_parameterization(self) -> None:
        cc = CountercurrentCNN(**small_kwargs())
        co = CocurrentCNN(**small_kwargs())
        self.assertEqual(parameter_counts(cc), parameter_counts(co))
        self.assertEqual(
            {key: value.shape for key, value in cc.state_dict().items()},
            {key: value.shape for key, value in co.state_dict().items()},
        )

    def test_countercurrent_and_cocurrent_have_identical_macs(self) -> None:
        sample = torch.zeros(1, 3, 32, 32)
        cc = CountercurrentCNN(**small_kwargs())
        co = CocurrentCNN(**small_kwargs())
        self.assertEqual(count_macs(cc, sample), count_macs(co, sample))

    def test_classic_alias_is_the_single_feedforward_control(self) -> None:
        model = ClassicCNN(channels=8, depth=2, num_groups=2)
        self.assertIsInstance(model, SingleStreamFeedForwardCNN)

    def test_primary_yaml_configs_differ_only_in_topology_identity(self) -> None:
        root = Path(__file__).parents[1] / "countercurrent_nn" / "configs"
        with (root / "cifar10_countercurrent_selfboundary.yaml").open(
            encoding="utf-8"
        ) as handle:
            cc = yaml.safe_load(handle)
        with (root / "cifar10_cocurrent_selfboundary.yaml").open(
            encoding="utf-8"
        ) as handle:
            co = yaml.safe_load(handle)
        self.assertEqual(cc["dataset"], co["dataset"])
        self.assertEqual(cc["training"], co["training"])
        cc_model = dict(cc["model"])
        co_model = dict(co["model"])
        cc_model.pop("name")
        co_model.pop("name")
        self.assertEqual(cc_model, co_model)


if __name__ == "__main__":
    unittest.main()
