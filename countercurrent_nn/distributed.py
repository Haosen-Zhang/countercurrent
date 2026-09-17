"""torchrun initialization; one process per GPU, with CPU/Gloo for smoke tests."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import distributed as dist

from .utils import resolve_device


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def initialize(requested_device: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(resolve_device(requested_device))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = resolve_device(requested_device)
    if device.type == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise ValueError("torchrun process count exceeds the visible GPU count")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif device.type == "cpu":
        backend = "gloo"
    else:
        raise ValueError("distributed training requires CUDA or CPU")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(device, rank, world_size)


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size <= 0 or global_batch_size % world_size:
        raise ValueError("global batch size must be positive and divisible by world size")
    return global_batch_size // world_size
