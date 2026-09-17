"""Four-process CPU/Gloo check: two DDP steps must match global-batch SGD.

Run with GLOO_SOCKET_IFNAME=lo and python -m torch.distributed.run
--nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29517
-m tests.distributed_ablation_smoke.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from countercurrent_nn.distributed import initialize
from countercurrent_nn.engine import build_optimizer, train_one_epoch
from countercurrent_nn.models import CocurrentCNN, CountercurrentCNN


def main():
    torch.set_num_threads(1)
    context = initialize("cpu")
    try:
        for cls in (CocurrentCNN, CountercurrentCNN):
            for variant in ("gamma_nodecay", "initial_aux", "v3"):
                torch.manual_seed(7)
                model = cls(channels=4, depth=2, refine_steps=3, num_groups=1, prototype_dim=4)
                reference = deepcopy(model)
                wrapped = DistributedDataParallel(model)
                config = dict(optimizer="sgd", lr=0.01, momentum=0.9, weight_decay=0.0005)
                if variant != "initial_aux":
                    config["conductance_weight_decay"] = 0.0
                weight = 0.0 if variant == "gamma_nodecay" else 0.2
                size = 2 * context.world_size
                batches = [(torch.randn(size, 3, 32, 32), torch.randint(10, (size,))) for _ in range(2)]
                shard = slice(2 * context.rank, 2 * context.rank + 2)
                local = [(x[shard], y[shard]) for x, y in batches]
                arguments = dict(global_step=0, total_steps=2, warmup_steps=0,
                                 base_lr=0.01, initial_loss_weight=weight)
                expected, _ = train_one_epoch(reference, batches, build_optimizer(reference, config),
                                              nn.CrossEntropyLoss(), context.device, **arguments)
                actual, step = train_one_epoch(wrapped, local, build_optimizer(wrapped, config),
                                               nn.CrossEntropyLoss(), context.device, **arguments)
                assert step == 2
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], rtol=2e-5, atol=2e-6)
                for key in ("loss", "final_loss", "accuracy", "examples", "initial_loss", "initial_accuracy"):
                    if key in expected:
                        assert abs(actual[key] - expected[key]) < 2e-6, (key, actual, expected)
                if context.is_main:
                    print(f"PASS {cls.__name__} {variant}: {context.world_size} ranks, "
                          "two steps and metrics match global-batch reference", flush=True)
                context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
