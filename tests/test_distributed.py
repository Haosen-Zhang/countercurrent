from __future__ import annotations

import unittest
from unittest.mock import patch

from torch.utils.data.distributed import DistributedSampler

from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.distributed import local_batch_size
from countercurrent_nn.train import _apply_overrides, parse_args


class DistributedDataTests(unittest.TestCase):
    def make_data(self, rank):
        return build_dataloaders(
            {"name": "cifar10", "batch_size": 2}, seed=0, synthetic=True,
            synthetic_train_size=16, synthetic_test_size=7, rank=rank, world_size=4,
        )

    def test_training_shards_are_disjoint_and_cover_the_dataset(self):
        shards = [set(self.make_data(rank).train.sampler) for rank in range(4)]
        self.assertEqual(set.union(*shards), set(range(16)))
        self.assertEqual(sum(map(len, shards)), 16)

    def test_evaluation_keeps_full_split_without_distributed_padding(self):
        data = self.make_data(0)
        self.assertNotIsInstance(data.test.sampler, DistributedSampler)
        self.assertEqual(sum(len(labels) for _, labels in data.test), 7)

    def test_training_sampler_changes_shuffle_each_epoch(self):
        sampler = self.make_data(0).train.sampler
        first = list(sampler)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_global_batch_is_split_across_four_ranks(self):
        self.assertEqual(local_batch_size(128, 4), 32)
        self.assertEqual(local_batch_size(128, 1), 128)
        for batch in (0, -4, 127):
            with self.assertRaises(ValueError):
                local_batch_size(batch, 4)

    def test_data_and_batch_cli_overrides_are_saved_in_config(self):
        with patch("sys.argv", [
            "train", "--config", "unused.yaml", "--data-root", "./dataset",
            "--no-download", "--batch-size", "128", "--num-workers", "2",
        ]):
            args = parse_args()
        config = {"dataset": {}, "training": {}, "runtime": {}}
        _apply_overrides(config, args)
        self.assertEqual(config["dataset"], {"root": "dataset", "download": False, "num_workers": 2})
        self.assertEqual(config["training"]["batch_size"], 128)


if __name__ == "__main__":
    unittest.main()
