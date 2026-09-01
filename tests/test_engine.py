from __future__ import annotations

import unittest

import torch
from torch import nn

from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.engine import (
    build_optimizer,
    evaluate,
    evaluate_refinement,
    train_one_epoch,
)
from countercurrent_nn.models import CountercurrentCNN, SingleStreamFeedForwardCNN


torch.set_num_threads(1)


class EngineSmokeTests(unittest.TestCase):
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
