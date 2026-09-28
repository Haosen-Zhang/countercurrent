"""Tests for the two-point boundary-value counterflow solver (plan P0/P1)."""

from __future__ import annotations

import unittest

import torch

from countercurrent_nn.models import build_model
from countercurrent_nn.models.boundary import InputConditionalTargetBoundary
from countercurrent_nn.models.invertible_transport import (
    ExactInverseTransport,
    InvertibleTransportBlock,
)

MODEL_KWARGS = dict(
    channels=32,
    depth=3,
    solve_steps=4,
    num_classes=10,
    num_groups=8,
    lattice_size=8,
    diagnostic_stride=1,
)


def tiny_model(name: str, **overrides):
    kwargs = {**MODEL_KWARGS, **overrides}
    return build_model(name, **kwargs).eval()


class InvertibleTransportTests(unittest.TestCase):
    def test_inverse_recovers_the_input(self):
        block = InvertibleTransportBlock(32, 8, beta=0.1).eval()
        block.residual[-1]._update_estimates()
        x = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            recovered = block.inverse(block(x))
        relative = (recovered - x).norm() / x.norm()
        self.assertLess(float(relative), 1e-5)

    def test_inverse_keeps_identity_jacobian_term(self):
        block = InvertibleTransportBlock(32, 8, beta=0.1)
        y = torch.randn(2, 32, 8, 8, requires_grad=True)
        gradient = torch.autograd.grad(block.inverse(y).sum(), y)[0]
        # d/dy (y - beta f(x*)) has an identity term, so the mean is near 1.
        self.assertGreater(float(gradient.mean()), 0.5)

    def test_exact_inverse_reuses_forward_weights(self):
        block = InvertibleTransportBlock(32, 8, beta=0.1).eval()
        block.residual[-1]._update_estimates()
        adapter = ExactInverseTransport(block)
        # the adapter owns no parameter of its own: every tensor it exposes is
        # literally the forward block's own tensor
        self.assertEqual(
            [id(p) for p in adapter.parameters()], [id(p) for p in block.parameters()]
        )
        x = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(adapter(block(x)), x, atol=1e-5))

    def test_rejects_beta_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            InvertibleTransportBlock(32, 8, beta=1.5)


class BoundaryTests(unittest.TestCase):
    def test_instance_term_is_conditioned_on_the_evidence(self):
        boundary = InputConditionalTargetBoundary(num_classes=10, channels=32)
        logits = torch.randn(2, 10)
        first = torch.randn(2, 32, 8, 8)
        second = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            out_first, _ = boundary(logits, first)
            out_second, _ = boundary(logits, second)
        self.assertFalse(torch.allclose(out_first, out_second))

    def test_stop_gradient_blocks_the_collapse_shortcut(self):
        boundary = InputConditionalTargetBoundary(num_classes=10, channels=32)
        with torch.no_grad():
            boundary.instance_scale.fill_(0.5)
        logits = torch.randn(2, 10)
        reference = torch.randn(2, 32, 8, 8, requires_grad=True)
        out, _ = boundary(logits, reference)
        gradient = torch.autograd.grad(out.sum(), reference, allow_unused=True)[0]
        self.assertIsNone(gradient)

    def test_gain_calibration_matches_the_target_magnitude(self):
        boundary = InputConditionalTargetBoundary(
            num_classes=10, channels=32, target_rms=2.5
        )
        reference = torch.zeros(1, 32, 8, 8)
        with torch.no_grad():
            out, _ = boundary(torch.zeros(1, 10), reference)
        self.assertAlmostEqual(float(out.square().mean().sqrt()), 2.5, places=3)

    def test_gain_stays_learnable(self):
        boundary = InputConditionalTargetBoundary(
            num_classes=10, channels=32, target_rms=2.5
        )
        logits = torch.zeros(1, 10, requires_grad=True)
        out, _ = boundary(logits, torch.zeros(1, 32, 8, 8))
        out.sum().backward()
        self.assertIsNotNone(boundary.gain.grad)
        self.assertNotEqual(float(boundary.gain.grad), 0.0)

    def test_uncalibrated_gain_is_the_identity(self):
        boundary = InputConditionalTargetBoundary(num_classes=10, channels=32)
        self.assertEqual(float(boundary.gain), 1.0)

    def test_hypothesis_changes_the_boundary_shape(self):
        boundary = InputConditionalTargetBoundary(num_classes=10, channels=32)
        reference = torch.zeros(1, 32, 8, 8)
        with torch.no_grad():
            hot, _ = boundary(torch.tensor([[8.0] + [0.0] * 9]), reference)
            cold, _ = boundary(torch.tensor([[0.0] * 9 + [8.0]]), reference)
        self.assertFalse(torch.allclose(hot, cold))


class SolverTests(unittest.TestCase):
    def test_solver_converges_and_reports_a_residual(self):
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, history = model.run_solver(x)
        self.assertGreater(len(history), 1)
        self.assertLess(
            history[-1]["relative_state_change"], history[0]["relative_state_change"]
        )

    def test_only_the_persistent_state_gates_convergence(self):
        # C is rebuilt from the target inlet every sweep, so its step-to-step
        # change must not be what stops the solver.
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, history = model.run_solver(x)
        final = history[-1]
        self.assertLess(final["relative_state_change"], 1e-3)
        self.assertGreater(final["relative_c_change"], 1e-3)

    def test_solution_is_a_fixed_point_of_the_cycle(self):
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            source = model.stem(x)
            states = model._pure_forward(source)
            logits = model._classify(states[-1])
            for _ in range(model.solve_steps):
                boundary = model._boundary_for(logits, states[-1])
                inlet = model._transport_c(boundary)
                states, _, _, _ = model._sweep(source, states, inlet, reverse_off=False)
                logits = model._classify(states[-1])
            # one more sweep from the converged state must barely move it
            boundary = model._boundary_for(logits, states[-1])
            inlet = model._transport_c(boundary)
            again, _, _, _ = model._sweep(source, states, inlet, reverse_off=False)
            relative = sum(
                (new - old).square().sum()
                for old, new in zip(states, again, strict=True)
            ).sqrt() / sum(state.square().sum() for state in states).sqrt()
        self.assertLess(float(relative), 1e-3)

    def test_countercurrent_and_cocurrent_transport_opposite_ways(self):
        counter = tiny_model("bvp_countercurrent")
        co = tiny_model("bvp_cocurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            boundary = torch.randn(2, 32, 8, 8)
            counter_states = counter._transport_c(boundary)
            co_states = co._transport_c(boundary)
        # the inlet is clamped at the opposite end of the depth axis
        self.assertTrue(torch.allclose(counter_states[-1], boundary))
        self.assertTrue(torch.allclose(co_states[0], boundary))
        self.assertFalse(torch.allclose(counter_states[0], co_states[0]))

    def test_reverse_off_changes_the_prediction(self):
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            full = model(x)
            off = model(x, reverse_off=True)
        self.assertGreater(float((full - off).norm() / full.norm()), 1e-3)

    def test_source_boundary_is_reinjected_every_sweep(self):
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, diagnostics = model(x, return_diagnostics=True)
        source = model.stem(x)
        for step in range(diagnostics["H_history"].shape[0]):
            self.assertTrue(
                torch.allclose(diagnostics["H_history"][step, 0], source, atol=1e-6)
            )

    def test_prediction_trajectory_is_padded_after_early_stopping(self):
        model = tiny_model(
            "bvp_countercurrent", solve_steps=8, tol=1e9
        )
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, history = model.run_solver(x)
            predictions = model.predict_iterations(x)
        self.assertEqual(len(history), 1)
        self.assertEqual(predictions.shape, (model.solve_steps + 1, 2, 10))
        for step in range(2, model.solve_steps + 1):
            torch.testing.assert_close(predictions[step], predictions[1])


class ContractTests(unittest.TestCase):
    def test_main_model_never_accepts_labels(self):
        model = tiny_model("bvp_countercurrent")
        x = torch.randn(2, 3, 32, 32)
        labels = torch.zeros(2, dtype=torch.long)
        with self.assertRaises(TypeError):
            model(x, labels)  # type: ignore[misc]

    def test_gradients_reach_every_parameter(self):
        for name in ("bvp_countercurrent", "inv_bvp_countercurrent"):
            model = tiny_model(name)
            x = torch.randn(4, 3, 32, 32)
            labels = torch.randint(0, 10, (4,))
            loss = torch.nn.functional.cross_entropy(model(x), labels)
            loss.backward()
            missing = [
                key
                for key, value in model.named_parameters()
                if value.grad is None or not torch.isfinite(value.grad).all()
            ]
            self.assertEqual(missing, [], f"{name} has missing gradients: {missing}")

    def test_invertible_flow_shares_the_reverse_weights(self):
        model = tiny_model("inv_bvp_countercurrent")
        forward_params = sum(p.numel() for p in model.F.parameters())
        reverse_params = sum(p.numel() for p in model.G.parameters())
        self.assertEqual(reverse_params, forward_params)
        forward_ids = [id(p) for block in model.F for p in block.parameters()]
        reverse_ids = [id(p) for block in model.G for p in block.parameters()]
        self.assertEqual(sorted(reverse_ids), sorted(forward_ids))


if __name__ == "__main__":
    unittest.main()
