from __future__ import annotations

import inspect
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

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
    build_model,
)
from countercurrent_nn.analysis.oracle_boundary import GTOracleCountercurrent
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


class ReciprocalInferenceRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.x = torch.randn(2, 3, 32, 32)

    def test_both_exchange_stages_are_paired_in_the_actual_network(self) -> None:
        for cls in (CountercurrentCNN, CocurrentCNN):
            model = cls(**small_kwargs()).eval()
            _, diag = model(self.x, return_diagnostics=True)
            for layer in range(model.depth):
                c_index = model._reverse_reference_index(layer)
                h_bar = diag["H_proposal"][:, layer + 1]
                c_bar = diag["C_proposal"][:, c_index]
                torch.testing.assert_close(
                    diag["H_reconciled"][:, layer + 1] + diag["C_reconciled"][:, c_index],
                    h_bar + c_bar,
                )
                torch.testing.assert_close(diag["proposal_D"][:, layer], h_bar - c_bar)
                torch.testing.assert_close(
                    diag["proposal_Q"][:, layer], model.exchange[layer].gamma * (h_bar - c_bar)
                )
                h_bar = diag["resweep_H_bar"][:, layer]
                c_bar = diag["C_reconciled"][:, c_index]
                torch.testing.assert_close(
                    diag["H_history"][1:, layer + 1] + diag["C_history"][:, c_index],
                    h_bar + c_bar,
                )
                torch.testing.assert_close(diag["D"][:, layer], h_bar - c_bar)
                torch.testing.assert_close(
                    diag["Q"][:, layer], model.exchange[layer].gamma * (h_bar - c_bar)
                )

    def test_corrected_reverse_state_propagates_through_the_next_g_block(self) -> None:
        for cls in (CountercurrentCNN, CocurrentCNN):
            model = cls(**small_kwargs()).eval()
            source = model.stem(self.x)
            boundary = torch.randn_like(source)
            _, _, trace = model._refine(source, boundary, reverse_off=False)
            if model.topology == "countercurrent":
                pairs = [
                    (trace["C_proposal"][position - 1], trace["C_reconciled"][position])
                    for position in range(1, model.depth)
                ]
            else:
                pairs = [
                    (trace["C_proposal"][position + 2], trace["C_reconciled"][position + 1])
                    for position in range(model.depth - 1)
                ]
            for transported, corrected_inlet in pairs:
                gradient = torch.autograd.grad(
                    transported.square().sum(), corrected_inlet, retain_graph=True
                )[0]
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_version_two_checkpoint_retains_the_old_schedule(self) -> None:
        current = CountercurrentCNN(**small_kwargs()).eval()
        old_schedule = deepcopy(current)
        old_schedule._inference_version.fill_(2)
        expected = old_schedule(self.x)
        restored = CountercurrentCNN(**small_kwargs()).eval()
        restored.load_state_dict(old_schedule.state_dict())
        self.assertEqual(restored._inference_version.item(), 2)
        torch.testing.assert_close(restored(self.x), expected)
        self.assertGreater((current(self.x) - expected).abs().max().item(), 1e-7)

    def test_evidence_corrected_c_is_on_the_prediction_graph_at_every_layer(self) -> None:
        for cls in (CountercurrentCNN, CocurrentCNN):
            model = cls(**small_kwargs()).eval()
            source = model.stem(self.x)
            # Hold the target fixed to isolate the local evidence -> C -> output path.
            boundary = torch.randn_like(source, requires_grad=True)
            states, _, trace = model._refine(source, boundary, reverse_off=False)
            loss = model._classify(states[-1]).square().sum()
            for layer in range(model.depth):
                corrected_c = trace["C_reconciled"][model._reverse_reference_index(layer)]
                h_proposal = trace["H_proposal"][layer + 1]
                h_to_c = torch.autograd.grad(
                    corrected_c.sum(), h_proposal, retain_graph=True
                )[0]
                c_to_prediction = torch.autograd.grad(
                    loss, corrected_c, retain_graph=True
                )[0]
                self.assertGreater(h_to_c.abs().sum().item(), 0)
                self.assertGreater(c_to_prediction.abs().sum().item(), 0)

    def test_removing_only_paired_c_update_changes_prediction_without_diagnostics(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        expected = model(self.x)
        exchange = model.exchange[0]
        original = exchange.paired_update

        def remove_c_correction(h_bar, c_bar):
            h, _, q = original(h_bar, c_bar)
            return h, c_bar, q

        with patch.object(exchange, "paired_update", side_effect=remove_c_correction):
            changed = model(self.x)
        self.assertGreater((expected - changed).abs().max().item(), 1e-7)

    def test_transport_direction_and_identical_initial_target_content(self) -> None:
        cc = CountercurrentCNN(**small_kwargs()).eval()
        co = CocurrentCNN(**small_kwargs()).eval()
        co.load_state_dict(cc.state_dict())
        boundaries = []
        for model, order in ((cc, [2, 1, 0]), (co, [0, 1, 2])):
            calls = []
            handles = [block.register_forward_hook(
                lambda _m, _i, _o, layer=layer: calls.append(layer)
            ) for layer, block in enumerate(model.G)]
            try:
                _, diag = model(self.x, return_diagnostics=True)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(calls, order * model.refine_steps)
            boundaries.append(diag["boundary_history"][0])
        torch.testing.assert_close(*boundaries)

    def test_labels_cannot_enter_main_forward_and_oracle_is_explicit(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).eval()
        labels = torch.tensor([0, 1])
        with self.assertRaises(TypeError):
            model(self.x, labels)
        with self.assertRaises(TypeError):
            model(self.x, labels=labels)
        normal = model(self.x)
        oracle = GTOracleCountercurrent(model)
        _, diag = oracle(self.x, labels, return_diagnostics=True)
        boundary = model.target_boundary
        expected = boundary.projector(boundary.class_prototypes[labels])
        expected = expected[:, :, None, None].expand(-1, -1, 8, 8)
        torch.testing.assert_close(diag["boundary_history"][0], expected)
        self.assertTrue(diag["oracle_analysis_only"].item())
        torch.testing.assert_close(model(self.x), normal)

    def test_final_ce_reaches_all_parameter_groups_and_earlier_hypothesis(self) -> None:
        model = CountercurrentCNN(**small_kwargs()).train()
        captured = []
        handle = model.head.register_forward_hook(lambda _m, _i, output: captured.append(output))
        try:
            logits = model(self.x)
        finally:
            handle.remove()
        captured[0].retain_grad()
        torch.nn.functional.cross_entropy(logits, torch.tensor([2, 5])).backward()
        self.assertGreater(captured[0].grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_diagnostics_do_not_change_inference_or_training_gradients(self) -> None:
        model = CountercurrentCNN(**small_kwargs())
        logits = model(self.x)
        gradients = torch.autograd.grad(logits.square().sum(), tuple(model.parameters()))
        captured_logits, _ = model(self.x, return_diagnostics=True)
        captured_gradients = torch.autograd.grad(
            captured_logits.square().sum(), tuple(model.parameters())
        )
        torch.testing.assert_close(logits, captured_logits)
        for normal, captured in zip(gradients, captured_gradients, strict=True):
            torch.testing.assert_close(normal, captured)

    def test_reverse_off_disables_both_flux_stages_and_keeps_source(self) -> None:
        for cls in (CountercurrentCNN, CocurrentCNN):
            model = cls(**small_kwargs()).eval()
            _, diag = model(self.x, return_diagnostics=True, reverse_off=True)
            for key in ("Q", "proposal_Q"):
                self.assertEqual(torch.count_nonzero(diag[key]).item(), 0)
            for state in diag["H_history"]:
                torch.testing.assert_close(state, diag["H_history"][0])

    def test_active_schedule_rejects_legacy_checkpoint_silently_reused_as_new(self) -> None:
        from countercurrent_nn.legacy.old_wrong_countercurrent.models import CountercurrentCNN as OldCC
        old = OldCC(**small_kwargs())
        with self.assertRaisesRegex(RuntimeError, "_inference_version"):
            CountercurrentCNN(**small_kwargs()).load_state_dict(old.state_dict())

    def test_forward_recurrent_matches_transport_budget(self) -> None:
        kwargs = small_kwargs()
        cc = CountercurrentCNN(**kwargs)
        recurrent = ForwardOnlyRecurrentCNN(**kwargs)
        sample = self.x[:1]
        # Conv/Linear scope: only the boundary projector is absent in recurrence.
        difference = count_macs(cc, sample) - count_macs(recurrent, sample)
        self.assertEqual(difference, cc.refine_steps * 8 * 8)

    def test_every_active_config_builds_and_predicts(self) -> None:
        config_root = Path(__file__).parents[1] / "countercurrent_nn" / "configs"
        for path in config_root.glob("*.yaml"):
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                config["model"].update(channels=8, num_groups=2)
                model = build_model(config["model"]).eval()
                with torch.no_grad():
                    logits = model(self.x)
                self.assertEqual(logits.shape, (2, 10))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
