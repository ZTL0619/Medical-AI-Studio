import gc
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ssnet_final.data.factory import build_train_loader, set_loader_epoch
from ssnet_final.engine.checkpoint import load_checkpoint, save_checkpoint
from ssnet_final.engine.distributed import (
    barrier,
    broadcast_float,
    cleanup_distributed,
    initialize_distributed,
    reduce_mean_dict,
)
from ssnet_final.engine.evaluation import evaluate, write_metrics
from ssnet_final.engine.status import StatusWriter
from ssnet_final.method.objective import build_objective
from ssnet_final.models.factory import build_model


class TrainingStepModule(nn.Module):
    """Keep all trainable components inside the DDP forward graph.

    DNP's Neural Process head is called by the objective rather than by the
    segmentation backbone's inference forward. Wrapping this training step
    ensures DDP sees every trainable parameter used to produce the loss.
    """

    def __init__(self, model, objective):
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(self, images, labels, iteration):
        return self.objective(self.model, images, labels, iteration)


def set_deterministic(seed, rank=0):
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(config, model):
    if config.optimizer == "adam":
        return Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _autocast_context(config, device):
    if not config.mixed_precision or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _make_grad_scaler(config, device):
    enabled = bool(config.mixed_precision and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _base_model(training_step):
    return training_step.module.model if isinstance(training_step, DDP) else training_step.model


def _append_history(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _release_cuda_training_resources(device):
    """Best-effort release of CUDA allocator/context resources owned by a worker.

    This runs only after the training graph/model/optimizer references have been
    dropped.  It does not change the numerical training path; it only makes the
    worker relinquish memory before the UI marks the job completed.
    """
    gc.collect()
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_train(config):
    context = initialize_distributed(config)
    status = StatusWriter(config.output / "status.json") if context.is_main else None
    writer = None
    progress = None
    model = None
    objective = None
    optimizer = None
    training_step = None
    train_loader = None
    scaler = None
    images = labels = loss = parts = batch = eval_model = None
    iteration = 0
    best_score = 0.0
    started = time.time()
    completed_ok = False
    try:
        set_deterministic(config.seed, context.rank)
        if context.is_main:
            config.output.mkdir(parents=True, exist_ok=True)
            (config.output / "config.json").write_text(
                json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            status.write(
                state="starting",
                iteration=0,
                max_iterations=config.max_iterations,
                best_dice=0.0,
                world_size=context.world_size,
                parallel_mode="ddp" if context.enabled else "single",
                global_batch_size=config.batch_size * context.world_size,
                gpu_ids=list(config.gpu_ids),
            )

        model = build_model(config, training=True).to(context.device)
        if context.enabled and config.sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        objective = build_objective(config).to(context.device)
        optimizer = build_optimizer(config, model)

        best_initialized = False
        if config.resume is not None:
            iteration, best_score, _ = load_checkpoint(
                config.resume, model, optimizer, map_location=context.device, expected_config=config
            )
            best_initialized = (config.output / "best.pt").exists()

        training_step = TrainingStepModule(model, objective).to(context.device)
        if context.enabled:
            training_step = DDP(
                training_step,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )

        train_loader = build_train_loader(
            config,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )
        scaler = _make_grad_scaler(config, context.device)

        if context.is_main:
            writer = SummaryWriter(str(config.output / "tensorboard"))
            progress = tqdm(
                total=config.max_iterations,
                initial=iteration,
                desc=(
                    f"{config.model}/{config.feature_learning}/{config.semi_supervision} "
                    f"x{context.world_size}GPU"
                ),
            )
            status.update(state="running", iteration=iteration, best_dice=best_score)

        epoch = 0
        started = time.time()
        while iteration < config.max_iterations:
            set_loader_epoch(train_loader, epoch)
            epoch += 1
            for batch in train_loader:
                if iteration >= config.max_iterations:
                    break

                images = batch["image"].to(context.device, non_blocking=True)
                labels = batch["label"].to(context.device, non_blocking=True)
                training_step.train()
                optimizer.zero_grad(set_to_none=True)

                with _autocast_context(config, context.device):
                    loss, parts = training_step(images, labels, iteration)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                iteration += 1

                if config.optimizer == "sgd" and iteration % 2500 == 0:
                    lr = config.learning_rate * 0.1 ** (iteration // 2500)
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                should_log = iteration == 1 or iteration % config.log_interval == 0
                if should_log:
                    reduced = reduce_mean_dict(parts, context)
                    if context.is_main:
                        for name, value in reduced.items():
                            writer.add_scalar(f"loss/{name}", value, iteration)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                        elapsed = max(time.time() - started, 1e-6)
                        speed = iteration * config.global_batch_size / elapsed
                        _append_history(
                            config.output / "history.jsonl",
                            {"type": "train", "iteration": iteration, "losses": reduced, "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed},
                        )
                        status.update(
                            state="running",
                            iteration=iteration,
                            max_iterations=config.max_iterations,
                            progress=iteration / config.max_iterations,
                            losses=reduced,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            best_dice=best_score,
                            samples_per_second=speed,
                            elapsed_seconds=elapsed,
                        )
                        progress.set_postfix(
                            loss=f"{reduced.get('total', float(loss.item())):.4f}",
                            best=f"{best_score:.4f}",
                        )

                if iteration >= config.validation_start and iteration % config.validation_interval == 0:
                    barrier(context)
                    if context.is_main:
                        eval_model = _base_model(training_step)
                        metrics = evaluate(config, eval_model, split="test", save_predictions=False)
                        score = metrics["mean"]["dice"]
                        writer.add_scalar("validation/dice", score, iteration)
                        write_metrics(config.output / "validation_latest.json", metrics)
                        best_path = config.output / "best.pt"
                        best_updated = (not best_initialized) or (score > best_score)
                        if best_updated:
                            best_score = round(float(score), 4)
                            save_checkpoint(
                                best_path,
                                eval_model,
                                optimizer,
                                iteration,
                                best_score,
                                config,
                            )
                            best_initialized = True
                        writer.add_scalar("validation/is_best", float(best_updated), iteration)
                        _append_history(
                            config.output / "history.jsonl",
                            {
                                "type": "validation",
                                "iteration": iteration,
                                "metrics": metrics["mean"],
                                "best_dice": best_score,
                                "best_updated": best_updated,
                                "best_checkpoint": str(best_path),
                            },
                        )
                        status.update(
                            validation=metrics["mean"],
                            best_dice=best_score,
                            best_updated=best_updated,
                            best_checkpoint=str(best_path),
                            iteration=iteration,
                        )
                    best_score = broadcast_float(best_score, context)
                    barrier(context)

                if iteration % config.save_interval == 0 and context.is_main:
                    save_checkpoint(
                        config.output / "last.pt",
                        _base_model(training_step),
                        optimizer,
                        iteration,
                        best_score,
                        config,
                    )

                if context.is_main:
                    progress.update(1)

        barrier(context)
        if context.is_main:
            eval_model = _base_model(training_step)
            save_checkpoint(
                config.output / "last.pt", eval_model, optimizer, iteration, best_score, config
            )
            # Do not fabricate best.pt at training shutdown. best.pt must always
            # correspond to a completed validation result.
            # Do not advertise completion until GPU/data-loader resources have
            # actually been released.  This prevents a second job from starting
            # while the previous CUDA context is still resident.
            status.update(
                state="finalizing",
                iteration=iteration,
                max_iterations=config.max_iterations,
                progress=1.0,
                best_dice=best_score,
                elapsed_seconds=time.time() - started,
            )
        completed_ok = True
    except Exception as exc:
        if context.is_main and status is not None:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if progress is not None:
            progress.close()

        # Drop every CUDA-owning reference before empty_cache().  empty_cache()
        # alone cannot release tensors that are still referenced by Python.
        batch = images = labels = loss = parts = eval_model = None
        train_loader = None
        scaler = None
        training_step = None
        optimizer = None
        objective = None
        model = None
        _release_cuda_training_resources(context.device)
        cleanup_distributed(context)

        if completed_ok and context.is_main and status is not None:
            status.update(
                state="completed",
                iteration=iteration,
                max_iterations=config.max_iterations,
                progress=1.0,
                best_dice=best_score,
                elapsed_seconds=time.time() - started,
                resources_released=True,
            )

def run_test(config):
    config.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = build_model(config, training=False).to(device)
    load_checkpoint(config.checkpoint, model, map_location=device, expected_config=config)
    metrics = evaluate(config, model, split="test", save_predictions=config.save_predictions, postprocess_lcc=True)
    write_metrics(config.output / "metrics.json", metrics)
    print(json.dumps(metrics["mean"], indent=2))
