from __future__ import annotations

import unittest

import torch
from torch import nn

from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import (
    build_optimizer,
    evaluate,
    evaluate_refinement,
    summarize_v5_mechanisms,
    train_one_epoch,
)
from countercurrent_nn.models import CountercurrentCNN, SingleStreamFeedForwardCNN


torch.set_num_threads(1)


class EngineSmokeTests(unittest.TestCase):
    def test_v5_mechanism_summary_exposes_pilot_gates(self) -> None:
        diagnostics = {
            "equation_residual": torch.tensor([[0.2, 0.1], [0.02, 0.04]]),
            "converged": torch.tensor([True, False]),
            "boundary_component_ratio": torch.tensor([[0.5, 0.25, 0.1]]),
            "cos_d_u_h": torch.tensor([[0.2, -0.4], [0.8, -0.6]]),
        }
        summary = summarize_v5_mechanisms(diagnostics)
        self.assertAlmostEqual(summary["equation_residual_final_mean"], 0.03)
        self.assertEqual(summary["equation_residual_decreased_fraction"], 1.0)
        self.assertEqual(summary["convergence_rate"], 0.5)
        self.assertEqual(summary["class_boundary_ratio"], 0.25)
        self.assertAlmostEqual(summary["cos_d_u_h_abs_final_mean"], 0.7)

    def test_synthetic_train_and_evaluate(self) -> None:
        data = build_dataloaders(
            {"name": "cifar10", "batch_size": 4},
            seed=0,
            synthetic=True,
            synthetic_train_size=8,
            synthetic_test_size=4,
        )
        model = SingleStreamFeedForwardCNN(
            channels=4, depth=1, num_groups=1, num_classes=10
        )
        optimizer = build_optimizer(
            model, {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 1e-4}
        )
        criterion = nn.CrossEntropyLoss()
        train_metrics, global_step = train_one_epoch(
            model,
            data.train,
            optimizer,
            criterion,
            torch.device("cpu"),
            global_step=0,
            total_steps=1,
            warmup_steps=0,
            base_lr=3e-4,
            limit_batches=1,
        )
        test_metrics = evaluate(model, data.test, criterion, torch.device("cpu"))
        self.assertEqual(global_step, 1)
        self.assertGreater(train_metrics["loss"], 0.0)
        self.assertGreater(test_metrics["loss"], 0.0)

    def test_refinement_metrics_include_confirmation_bias_diagnostics(self) -> None:
        data = build_dataloaders(
            {"name": "cifar10", "batch_size": 4},
            seed=0,
            synthetic=True,
            synthetic_train_size=8,
            synthetic_test_size=4,
        )
        model = CountercurrentCNN(
            channels=4,
            depth=1,
            refine_steps=2,
            num_classes=10,
            num_groups=1,
            prototype_dim=4,
        )
        metrics = evaluate_refinement(
            model, data.test, torch.device("cpu"), limit_batches=1
        )
        self.assertEqual(len(metrics["iteration_accuracy"]), 3)
        self.assertEqual(len(metrics["iteration_entropy"]), 3)
        self.assertIn("error_correction_rate", metrics)
        self.assertIn("error_amplification_rate", metrics)


if __name__ == "__main__":
    unittest.main()
