"""CIFAR-10/100 data construction with the mandated light augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .utils import seed_worker

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader[Any]
    validation: DataLoader[Any] | None
    test: DataLoader[Any]
    num_classes: int


class SyntheticCIFAR(Dataset[tuple[Tensor, Tensor]]):
    """Deterministic in-memory data used only for smoke tests."""

    def __init__(self, size: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, 32, 32, generator=generator)
        self.targets = torch.randint(num_classes, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.targets[index]


def _torchvision_datasets_and_transforms() -> tuple[Any, Any]:
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required for real CIFAR data; install project requirements"
        ) from error
    return datasets, transforms


def _make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(
    config: dict[str, Any],
    *,
    seed: int,
    synthetic: bool = False,
    synthetic_train_size: int = 256,
    synthetic_test_size: int = 128,
) -> DataBundle:
    name = str(config.get("name", "cifar10")).lower()
    if name not in CIFAR_STATS:
        raise ValueError(f"dataset must be 'cifar10' or 'cifar100', got {name!r}")
    num_classes = 10 if name == "cifar10" else 100
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", True))

    if synthetic:
        train_dataset = SyntheticCIFAR(synthetic_train_size, num_classes, seed)
        test_dataset = SyntheticCIFAR(synthetic_test_size, num_classes, seed + 1)
        train_loader = _make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            seed=seed,
        )
        test_loader = _make_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            seed=seed + 1,
        )
        return DataBundle(train_loader, None, test_loader, num_classes)

    datasets, transforms = _torchvision_datasets_and_transforms()
    mean, std = CIFAR_STATS[name]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    dataset_type = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    root = Path(config.get("root", "./data")).expanduser()
    download = bool(config.get("download", True))
    train_augmented = dataset_type(
        root=str(root), train=True, transform=train_transform, download=download
    )
    train_evaluation = dataset_type(
        root=str(root), train=True, transform=evaluation_transform, download=download
    )
    test_dataset = dataset_type(
        root=str(root), train=False, transform=evaluation_transform, download=download
    )

    val_size = int(config.get("val_size", 5000))
    if not 0 <= val_size < len(train_augmented):
        raise ValueError(f"val_size must be in [0, {len(train_augmented) - 1}]")
    # Keep the formal 45k/5k partition fixed while model/loader seeds vary.
    split_seed = int(config.get("split_seed", 0))
    permutation = torch.randperm(
        len(train_augmented), generator=torch.Generator().manual_seed(split_seed)
    ).tolist()
    validation_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    train_dataset: Dataset[Any] = Subset(train_augmented, train_indices)
    validation_dataset: Dataset[Any] | None = (
        Subset(train_evaluation, validation_indices) if val_size else None
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    validation_loader = (
        _make_loader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 1,
        )
        if validation_dataset is not None
        else None
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed + 2,
    )
    return DataBundle(train_loader, validation_loader, test_loader, num_classes)
