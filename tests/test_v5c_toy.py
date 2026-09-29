"""Linear toy BVP gate required before CIFAR-10 V5-C training."""

from __future__ import annotations

import unittest

import torch


def linear_map(state, *, h0, c2, a=0.55, b=0.50, gamma=0.10):
    h1, h2, _c0, c1 = state.unbind()
    return torch.stack(
        [
            (1 - gamma) * a * h0 + gamma * b * c1,
            (1 - gamma) * a * h1 + gamma * b * c2,
            gamma * a * h0 + (1 - gamma) * b * c1,
            gamma * a * h1 + (1 - gamma) * b * c2,
        ]
    )


def direct_solution(*, h0, c2, a=0.55, b=0.50, gamma=0.10):
    zero = torch.zeros(4, dtype=torch.float64)
    offset = linear_map(zero, h0=h0, c2=c2, a=a, b=b, gamma=gamma)
    columns = []
    for index in range(4):
        basis = torch.zeros(4, dtype=torch.float64)
        basis[index] = 1
        columns.append(
            linear_map(basis, h0=h0, c2=c2, a=a, b=b, gamma=gamma) - offset
        )
    matrix = torch.stack(columns, dim=1)
    return torch.linalg.solve(torch.eye(4, dtype=torch.float64) - matrix, offset)


def iterate(damping, *, max_steps=256, tolerance=1e-10):
    state = torch.zeros(4, dtype=torch.float64)
    residuals = []
    for step in range(max_steps):
        candidate = linear_map(state, h0=1.2, c2=-0.4)
        residual = (candidate - state).norm() / state.norm().clamp_min(1e-12)
        residuals.append(float(residual))
        if residual < tolerance:
            return state, residuals, step + 1
        state = state + damping * (candidate - state)
    return state, residuals, max_steps


class LinearToyBVPTests(unittest.TestCase):
    def test_iterative_solution_matches_direct_linear_solve(self):
        expected = direct_solution(h0=1.2, c2=-0.4)
        actual, residuals, _ = iterate(0.5)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        self.assertLess(residuals[-1], residuals[0])

    def test_damping_changes_rate_but_not_fixed_point(self):
        slow, slow_residuals, slow_steps = iterate(0.25)
        fast, fast_residuals, fast_steps = iterate(0.75)
        torch.testing.assert_close(slow, fast, atol=1e-9, rtol=1e-9)
        self.assertNotEqual(slow_steps, fast_steps)
        self.assertLess(slow_residuals[-1], 1e-9)
        self.assertLess(fast_residuals[-1], 1e-9)

    def test_early_stop_agrees_with_long_fixed_iteration(self):
        stopped, _, _ = iterate(0.5, tolerance=1e-10)
        fixed, _, _ = iterate(0.5, max_steps=256, tolerance=0.0)
        torch.testing.assert_close(stopped, fixed, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
