"""Train Countercurrent CNN and its CIFAR controls with one shared protocol."""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from countercurrent_nn.config import dump_config, load_config, require_sections
from countercurrent_nn.data import build_dataloaders
from countercurrent_nn.distributed import DistributedContext, initialize, local_batch_size
from countercurrent_nn.engine import (
    build_optimizer,
    collect_diagnostics,
    evaluate,
    evaluate_refinement,
    summarize_diagnostics,
    train_one_epoch,
)
from countercurrent_nn.models import build_model
from countercurrent_nn.profiling import profile_model
from countercurrent_nn.utils import (
    append_jsonl,
    parameter_counts,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None, help="global batch size across all ranks")
    parser.add_argument("--num-workers", type=int, default=None, help="loader workers per rank")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="use the documented AdamW debug setup")
    parser.add_argument("--synthetic-data", action="store_true", help="smoke tests only")
    parser.add_argument("--synthetic-train-size", type=int, default=256)
    parser.add_argument("--synthetic-test-size", type=int, default=128)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config["training"]
    runtime = config["runtime"]
    if args.data_root is not None:
        config["dataset"]["root"] = str(args.data_root)
    if args.no_download:
        config["dataset"]["download"] = False
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["dataset"]["num_workers"] = args.num_workers
    if args.seed is not None:
        runtime["seed"] = args.seed
    if args.device is not None:
        runtime["device"] = args.device
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.output_dir is not None:
        runtime["output_dir"] = str(args.output_dir)
    if args.debug:
        training.update(
            {
                "optimizer": "adamw",
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "epochs": args.epochs if args.epochs is not None else 3,
                "warmup_epochs": 0,
            }
        )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "dataset", "model", "training", "runtime")
    _apply_overrides(config, args)
    context = initialize(str(config["runtime"].get("device", "auto")))
    try:
        _train(args, config, context)
    finally:
        context.close()


def _train(args: argparse.Namespace, config: dict[str, Any], context: DistributedContext) -> None:
    training = config["training"]
    runtime = config["runtime"]
    seed = int(runtime.get("seed", 0))
    set_seed(seed, deterministic=bool(runtime.get("deterministic", False)))
    device = context.device

    data_config = dict(config["dataset"])
    global_batch = int(training.get("batch_size", 128))
    data_config["batch_size"] = local_batch_size(global_batch, context.world_size)
    # When downloading is requested, let rank zero finish before others read.
    serialize_download = context.enabled and data_config.get("download", True) and not args.synthetic_data
    if serialize_download and not context.is_main:
        context.barrier()
        data_config["download"] = False
    data = build_dataloaders(
        data_config,
        seed=seed,
        synthetic=args.synthetic_data,
        synthetic_train_size=args.synthetic_train_size,
        synthetic_test_size=args.synthetic_test_size,
        rank=context.rank,
        world_size=context.world_size,
    )
    if serialize_download and context.is_main:
        context.barrier()
    model_config = dict(config["model"])
    model_config["num_classes"] = data.num_classes
    model = build_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, training)

    output_dir = Path(runtime.get("output_dir", "results/run"))
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.resume is None and any((output_dir / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl")):
            raise FileExistsError(f"existing experiment at {output_dir}; use a new output directory or --resume")
    metrics_path = output_dir / "metrics.jsonl"

    epochs = int(training.get("epochs", 200))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    effective_batches = len(data.train)
    if args.limit_train_batches is not None:
        effective_batches = min(effective_batches, args.limit_train_batches)
    total_steps = max(epochs * effective_batches, 1)
    warmup_steps = int(training.get("warmup_epochs", 5)) * effective_batches
    base_lr = float(training.get("lr", 0.1))
    amp = bool(training.get("amp", False))

    start_epoch = 0
    global_step = 0
    best_accuracy = float("-inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_version = checkpoint["model_state"].get("_inference_version")
        requested_version = getattr(model, "_inference_version", None)
        if (saved_version is not None and requested_version is not None
                and int(saved_version.item()) != int(requested_version.item())):
            raise ValueError(
                "cannot resume with a different inference_version; "
                "start a fresh experiment"
            )
        previous = checkpoint["config"]["training"]
        for key, default in (("initial_loss_weight", 0.0), ("conductance_weight_decay", None)):
            old_value = previous.get(key, previous.get("weight_decay", 5e-4) if default is None else default)
            new_value = training.get(key, training.get("weight_decay", 5e-4) if default is None else default)
            if float(old_value) != float(new_value):
                raise ValueError(f"cannot resume with a different {key}; start a fresh experiment")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", start_epoch * effective_batches))
        best_accuracy = float(checkpoint.get("best_accuracy", best_accuracy))

    if context.is_main:
        dump_config(config, output_dir / "config.yaml")

    training_model = model
    if context.enabled:
        training_model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=not getattr(model, "exchange_enabled", True),
        )
    if context.is_main:
        sample = torch.zeros(1, 3, 32, 32, device=device)
        architecture_profile = profile_model(model, sample, warmup_runs=0, measured_runs=1)
        write_json(architecture_profile, output_dir / "profile.json")
        print(
            f"model={model_config['name']} device={device} params={architecture_profile['params']:,} "
            f"MACs={architecture_profile['macs']:,} world_size={context.world_size} "
            f"global_batch={global_batch} per_rank_batch={data_config['batch_size']}", flush=True,
        )
    context.barrier()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    selection_loader = data.validation if data.validation is not None else data.test
    for epoch in range(start_epoch, epochs):
        if hasattr(data.train.sampler, "set_epoch"):
            data.train.sampler.set_epoch(epoch)
        train_metrics, global_step = train_one_epoch(
            training_model,
            data.train,
            optimizer,
            criterion,
            device,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            amp=amp,
            initial_loss_weight=float(training.get("initial_loss_weight", 0.0)),
            limit_batches=args.limit_train_batches,
        )
        # Evaluate the unwrapped model on rank zero over the complete split.
        # No DistributedSampler padding or duplicate validation/test examples.
        if not context.is_main:
            context.barrier()
            continue
        validation_metrics = evaluate(
            model,
            selection_loader,
            criterion,
            device,
            limit_batches=args.limit_eval_batches,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation" if data.validation is not None else "test": validation_metrics,
        }
        append_jsonl(row, metrics_path)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_acc={validation_metrics['accuracy']:.4f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_accuracy": max(best_accuracy, validation_metrics["accuracy"]),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "metrics": row,
        }
        _save_checkpoint(output_dir / "last.pt", checkpoint_payload)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            _save_checkpoint(output_dir / "best.pt", checkpoint_payload)
        context.barrier()

    # All ranks have finished synchronized training. Final evaluation is local.
    context.close()
    if not context.is_main:
        return

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        data.test,
        criterion,
        device,
        limit_batches=args.limit_eval_batches,
    )
    refinement_metrics = evaluate_refinement(
        model,
        data.test,
        device,
        limit_batches=args.limit_eval_batches,
    )
    supports_reverse_off = "reverse_off" in inspect.signature(model.forward).parameters
    reverse_off_metrics: dict[str, Any] = {}
    reverse_off_refinement: dict[str, Any] = {}
    if supports_reverse_off:
        reverse_off_metrics = evaluate(
            model,
            data.test,
            criterion,
            device,
            limit_batches=args.limit_eval_batches,
            forward_kwargs={"reverse_off": True},
        )
        reverse_off_refinement = evaluate_refinement(
            model,
            data.test,
            device,
            limit_batches=args.limit_eval_batches,
            reverse_off=True,
        )
    diagnostic_config = config.get("diagnostics", {})
    diagnostics = collect_diagnostics(
        model,
        data.test,
        device,
        max_samples=int(diagnostic_config.get("max_samples", 16)),
    )
    if diagnostics:
        torch.save(diagnostics, output_dir / "diagnostics.pt")
        write_json(summarize_diagnostics(diagnostics), output_dir / "diagnostics.json")

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary = {
        "model": model_config["name"],
        "inference_version": int(getattr(model, "_inference_version", torch.tensor(0)).item()),
        "seed": seed,
        "world_size": context.world_size,
        "global_batch_size": global_batch,
        "per_rank_batch_size": data_config["batch_size"],
        **parameter_counts(model),
        "macs": architecture_profile["macs"],
        "flops_approx": architecture_profile["flops_approx"],
        "best_selection_accuracy": best_accuracy,
        "test": test_metrics,
        "refinement": refinement_metrics,
        "reverse_off_test": reverse_off_metrics,
        "reverse_off_refinement": reverse_off_refinement,
        "reverse_accuracy_drop": (
            test_metrics["accuracy"] - reverse_off_metrics["accuracy"]
            if reverse_off_metrics
            else None
        ),
        "wall_time_seconds": time.perf_counter() - wall_start,
        "peak_memory_bytes": peak_memory,
        "synthetic_data": args.synthetic_data,
    }
    write_json(summary, output_dir / "summary.json")
    print(f"test_acc={test_metrics['accuracy']:.4f} output={output_dir}")


if __name__ == "__main__":
    main()
