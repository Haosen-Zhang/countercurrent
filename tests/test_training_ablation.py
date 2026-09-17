from copy import deepcopy
from pathlib import Path
import unittest

import torch
from torch import nn

from countercurrent_nn.config import load_config
from countercurrent_nn.engine import build_optimizer, train_one_epoch
from countercurrent_nn.models import CocurrentCNN, CountercurrentCNN, SingleStreamFeedForwardCNN


torch.set_num_threads(1)


def small_model(cls=CountercurrentCNN):
    return cls(channels=4, depth=2, refine_steps=3, num_groups=1, prototype_dim=4)


class TrainingAblationTests(unittest.TestCase):
    def test_zero_task_gradient_preserves_only_exempt_conductance(self):
        for name in ("sgd", "adamw"):
            for exempt in (False, True):
                with self.subTest(optimizer=name, exempt=exempt):
                    model = small_model()
                    config = dict(optimizer=name, lr=0.1, momentum=0.9, weight_decay=0.1)
                    if exempt:
                        config["conductance_weight_decay"] = 0.0
                    optimizer = build_optimizer(model, config)
                    grouped = [p for group in optimizer.param_groups for p in group["params"]]
                    self.assertEqual(len(grouped), len(set(map(id, grouped))))
                    self.assertEqual(set(map(id, grouped)), set(map(id, model.parameters())))
                    gamma_before = model.exchange[0].gamma.detach().clone()
                    ordinary_before = model.head.weight.detach().clone()
                    for parameter in model.parameters():
                        parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                    if exempt:
                        torch.testing.assert_close(model.exchange[0].gamma, gamma_before, rtol=0, atol=0)
                    else:
                        self.assertTrue((model.exchange[0].gamma > gamma_before).all())
                    self.assertFalse(torch.equal(model.head.weight, ordinary_before))

    def test_auxiliary_loss_gradients_match_manual_objective_without_extra_sweeps(self):
        torch.manual_seed(10)
        images, targets = torch.randn(2, 3, 32, 32), torch.tensor([2, 5])
        for cls in (CountercurrentCNN, CocurrentCNN):
            for weight in (0.0, 0.2):
                with self.subTest(model=cls.__name__, weight=weight):
                    model = small_model(cls)
                    reference = deepcopy(model)
                    steps = reference.predict_iterations(images)
                    final_ce = nn.functional.cross_entropy(steps[-1], targets)
                    initial_ce = nn.functional.cross_entropy(steps[0], targets)
                    expected = final_ce + weight * initial_ce
                    expected.backward()
                    calls = []
                    handle = model.stem.register_forward_hook(lambda *_: calls.append(1))
                    try:
                        metrics, step = train_one_epoch(
                            model, [(images, targets)], build_optimizer(model, {"lr": 0.0}),
                            nn.CrossEntropyLoss(), torch.device("cpu"), global_step=0,
                            total_steps=1, warmup_steps=0, base_lr=0.0,
                            initial_loss_weight=weight,
                        )
                    finally:
                        handle.remove()
                    self.assertEqual(step, 1)
                    self.assertEqual(len(calls), model.refine_steps + 1)
                    self.assertAlmostEqual(metrics["loss"], expected.item(), places=6)
                    self.assertAlmostEqual(metrics["final_loss"], final_ce.item(), places=6)
                    self.assertEqual("initial_loss" in metrics, bool(weight))
                    if weight:
                        self.assertAlmostEqual(metrics["initial_loss"], initial_ce.item(), places=6)
                    for actual, wanted in zip(model.parameters(), reference.parameters()):
                        self.assertIsNotNone(actual.grad)
                        self.assertTrue(torch.isfinite(actual.grad).all())
                        torch.testing.assert_close(actual.grad, wanted.grad)

    def test_initial_output_is_differentiable_and_keeps_default_inference(self):
        model = small_model()
        images = torch.randn(2, 3, 32, 32)
        expected = model(images)
        final, initial = model(images, return_initial_logits=True)
        torch.testing.assert_close(final, expected, atol=0, rtol=0)
        self.assertTrue(initial.requires_grad)
        with self.assertRaises(ValueError):
            model(images, return_initial_logits=True, return_diagnostics=True)

    def test_invalid_training_options_fail_clearly(self):
        model = small_model()
        for value in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_optimizer(model, {"conductance_weight_decay": value})
            with self.assertRaises(ValueError):
                train_one_epoch(model, [], build_optimizer(model, {}), nn.CrossEntropyLoss(),
                                torch.device("cpu"), global_step=0, total_steps=1,
                                warmup_steps=0, base_lr=0.1, initial_loss_weight=value)
        ordinary = SingleStreamFeedForwardCNN(channels=4, depth=1, num_groups=1)
        with self.assertRaises(ValueError):
            build_optimizer(ordinary, {"conductance_weight_decay": 0.0})
        with self.assertRaises(ValueError):
            train_one_epoch(ordinary, [], build_optimizer(ordinary, {}), nn.CrossEntropyLoss(),
                            torch.device("cpu"), global_step=0, total_steps=1,
                            warmup_steps=0, base_lr=0.1, initial_loss_weight=0.2)

    def test_ablation_configs_keep_the_original_protocol_and_pair_topologies(self):
        root = Path(__file__).parents[1] / "countercurrent_nn/configs"
        original = load_config(root / "cifar10_countercurrent_selfboundary.yaml")
        for variant in ("gamma_nodecay", "initial_aux", "v3"):
            cc = load_config(root / f"cifar10_countercurrent_{variant}.yaml")
            co = load_config(root / f"cifar10_cocurrent_{variant}.yaml")
            self.assertEqual(cc["training"], co["training"])
            self.assertEqual(cc["dataset"], co["dataset"])
            self.assertEqual(cc["model"], dict(co["model"], name="countercurrent"))
            self.assertEqual(cc["model"], original["model"])
            for key, value in original["training"].items():
                self.assertEqual(cc["training"][key], value)
            self.assertEqual(cc["training"]["initial_loss_weight"], 0.0 if variant == "gamma_nodecay" else 0.2)
            self.assertEqual(cc["training"].get("conductance_weight_decay", 0.0005),
                             0.0005 if variant == "initial_aux" else 0.0)


if __name__ == "__main__":
    unittest.main()
