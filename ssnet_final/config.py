"""Configuration for Medical Image Segmentation Studio.

The training system is modular: backbone, optional FSL feature learning, and
semi-supervision strategy are selected independently.  Runtime/distributed
settings are part of the same serializable configuration so the UI and CLI launch exactly the same training engine.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple


MODEL_ALIASES = {
    "vnet": "vnet",
    "catnet": "catnet3d",
    "catnet3d": "catnet3d",
    "3dcatnet": "catnet3d",
}

FEATURE_ALIASES = {
    "none": "none",
    "off": "none",
    "fsl": "fsl",
}

SEMI_ALIASES = {
    "none": "none",
    "off": "none",
    "traditional": "traditional",
    "mutual": "traditional",
    "fsl": "traditional",
    "dnp": "dnp",
}

LABELED_COUNTS = {5: 4, 10: 8, 100: 80}


@dataclass(frozen=True)
class ExperimentConfig:
    command: str
    experiment_name: str
    dataset: str
    model: str
    feature_learning: str
    semi_supervision: str
    labeled_percent: int
    data_root: Path
    output: Path
    labeled: int
    device: str
    seed: int
    workers: int
    num_classes: int
    patch_size: Tuple[int, int, int]
    batch_size: int
    labeled_batch_size: int
    max_iterations: int
    learning_rate: float
    momentum: float
    weight_decay: float
    optimizer: str
    tau: float
    sharpening_temperature: float
    np_samples: int
    dnp_segmentation_weight: float
    dnp_contrastive_weight: float
    dnp_consistency: float
    dnp_consistency_rampup: float
    dnp_lds_weight: float
    dnp_distribution_weight: float
    dnp_np_supervised_weight: float
    dnp_vat_magnitude: float
    anchor_temperature: float
    beta: float
    fuzziness_weight: float
    validation_interval: int
    validation_start: int
    save_interval: int
    log_interval: int
    parallel_mode: str
    gpu_ids: Tuple[int, ...]
    mixed_precision: bool
    sync_batchnorm: bool
    linear_lr_scaling: bool
    ddp_find_unused_parameters: bool
    resume: Optional[Path] = None
    checkpoint: Optional[Path] = None
    save_predictions: bool = False

    @property
    def world_size(self):
        return len(self.gpu_ids) if self.parallel_mode == "ddp" else 1

    @property
    def global_batch_size(self):
        return self.batch_size * self.world_size

    @property
    def global_labeled_batch_size(self):
        return self.labeled_batch_size * self.world_size

    def to_json_dict(self):
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                values[key] = str(value)
            elif isinstance(value, tuple):
                values[key] = list(value)
        return values


def _normalize(value, aliases, label):
    key = str(value).lower()
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported {label}: {value}. Valid values: {valid}")
    return aliases[key]


def build_config(
    command,
    dataset,
    model,
    feature_learning,
    semi_supervision,
    labeled_percent,
    data_root,
    output,
    device="cuda:0",
    experiment_name="",
    seed=1337,
    workers=4,
    validation_interval=200,
    save_interval=1000,
    log_interval=10,
    resume=None,
    checkpoint=None,
    save_predictions=False,
    anchor_temperature=100.0,
    parallel_mode="single",
    gpu_ids=(0,),
    mixed_precision=False,
    sync_batchnorm=False,
    linear_lr_scaling=False,
    ddp_find_unused_parameters=False,
    batch_size=None,
    labeled_batch_size=None,
    max_iterations=None,
    learning_rate=None,
):
    command = str(command).lower()
    dataset = str(dataset).upper()
    model = _normalize(model, MODEL_ALIASES, "model")
    feature_learning = _normalize(feature_learning, FEATURE_ALIASES, "feature learning")
    semi_supervision = _normalize(semi_supervision, SEMI_ALIASES, "semi-supervision")
    labeled_percent = int(labeled_percent)
    parallel_mode = str(parallel_mode).lower()
    gpu_ids = tuple(int(item) for item in gpu_ids)

    if command not in {"train", "test"}:
        raise ValueError("command must be 'train' or 'test'")
    if dataset != "LA":
        raise ValueError("This project is 3-D only and currently supports the 3-D LA dataset.")
    if labeled_percent not in LABELED_COUNTS:
        raise ValueError("labeled_percent must be 5, 10, or 100 for LA.")
    if semi_supervision != "none" and model != "vnet":
        raise ValueError(
            "Traditional and DNP semi-supervision require the dual-decoder V-Net backbone."
        )
    if semi_supervision != "none" and labeled_percent == 100:
        raise ValueError("Semi-supervision requires unlabeled data; use 5% or 10% labeled data.")
    if feature_learning == "fsl" and model != "vnet":
        raise ValueError("The integrated FSL feature module is attached to the V-Net bottleneck.")
    if feature_learning == "fsl" and labeled_percent not in {5, 10}:
        raise ValueError("The integrated FSL settings are defined for 5% or 10% labeled LA data.")
    if parallel_mode not in {"single", "ddp"}:
        raise ValueError("parallel_mode must be 'single' or 'ddp'.")
    if not gpu_ids:
        gpu_ids = (0,)
    if parallel_mode == "ddp" and len(gpu_ids) < 2:
        raise ValueError("DDP mode requires at least two GPU IDs.")

    labeled = LABELED_COUNTS[labeled_percent]
    num_classes = 2
    patch_size = (112, 112, 80)

    if model == "catnet3d":
        default_batch_size = 1
        default_labeled_batch_size = 1
        default_max_iterations = 40000
        default_learning_rate = 1e-4
        momentum = 0.0
        weight_decay = 0.0
        optimizer = "adam"
        validation_start = 0
    else:
        default_batch_size = 4 if semi_supervision != "none" else 2
        default_labeled_batch_size = 2 if semi_supervision != "none" else default_batch_size
        default_max_iterations = 15000 if semi_supervision == "dnp" and labeled_percent == 5 else 10000
        default_learning_rate = 0.01
        momentum = 0.9
        weight_decay = 0.0001
        optimizer = "sgd"
        validation_start = 2000

    batch_size = int(default_batch_size if batch_size is None else batch_size)
    labeled_batch_size = int(
        default_labeled_batch_size if labeled_batch_size is None else labeled_batch_size
    )
    max_iterations = int(default_max_iterations if max_iterations is None else max_iterations)
    learning_rate = float(default_learning_rate if learning_rate is None else learning_rate)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if semi_supervision != "none":
        if not 0 < labeled_batch_size < batch_size:
            raise ValueError(
                "For semi-supervised training each GPU batch must contain both labeled and unlabeled samples."
            )
    else:
        labeled_batch_size = batch_size

    world_size = len(gpu_ids) if parallel_mode == "ddp" else 1
    if linear_lr_scaling and world_size > 1:
        learning_rate *= world_size

    if semi_supervision == "dnp":
        tau = 0.5  # CCAI aux4: --auxf default=0.5
        sharpening_temperature = 0.4
        np_samples = 3
        # Exact weights from the user-selected aux4 objective:
        # 0.5 * (Dice + CE + loss_consist)
        # + w(t) * [loss_lds + 0.05 * (CL_l + CL_u)
        #           + 0.01 * WSD + 0.1 * NP_sup].
        dnp_segmentation_weight = 0.5
        dnp_contrastive_weight = 0.05
        dnp_consistency = 1.0
        dnp_consistency_rampup = 40.0
        dnp_lds_weight = 1.0
        dnp_distribution_weight = 0.01
        dnp_np_supervised_weight = 0.1
        dnp_vat_magnitude = 10.0
    elif semi_supervision == "traditional":
        tau = 0.0
        sharpening_temperature = 0.7
        np_samples = 0
        dnp_segmentation_weight = 0.0
        dnp_contrastive_weight = 0.0
        dnp_consistency = 0.0
        dnp_consistency_rampup = 0.0
        dnp_lds_weight = 0.0
        dnp_distribution_weight = 0.0
        dnp_np_supervised_weight = 0.0
        dnp_vat_magnitude = 0.0
    else:
        tau = 0.0
        sharpening_temperature = 1.0
        np_samples = 0
        dnp_segmentation_weight = 0.0
        dnp_contrastive_weight = 0.0
        dnp_consistency = 0.0
        dnp_consistency_rampup = 0.0
        dnp_lds_weight = 0.0
        dnp_distribution_weight = 0.0
        dnp_np_supervised_weight = 0.0
        dnp_vat_magnitude = 0.0

    beta = {5: 0.8, 10: 0.5}.get(labeled_percent, 0.0) if feature_learning == "fsl" else 0.0
    fuzziness_weight = 0.05 if feature_learning == "fsl" else 0.0

    return ExperimentConfig(
        command=command,
        experiment_name=str(experiment_name),
        dataset=dataset,
        model=model,
        feature_learning=feature_learning,
        semi_supervision=semi_supervision,
        labeled_percent=labeled_percent,
        data_root=Path(data_root),
        output=Path(output),
        labeled=labeled,
        device=str(device),
        seed=int(seed),
        workers=int(workers),
        num_classes=num_classes,
        patch_size=patch_size,
        batch_size=batch_size,
        labeled_batch_size=labeled_batch_size,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        tau=tau,
        sharpening_temperature=sharpening_temperature,
        np_samples=np_samples,
        dnp_segmentation_weight=dnp_segmentation_weight,
        dnp_contrastive_weight=dnp_contrastive_weight,
        dnp_consistency=dnp_consistency,
        dnp_consistency_rampup=dnp_consistency_rampup,
        dnp_lds_weight=dnp_lds_weight,
        dnp_distribution_weight=dnp_distribution_weight,
        dnp_np_supervised_weight=dnp_np_supervised_weight,
        dnp_vat_magnitude=dnp_vat_magnitude,
        anchor_temperature=float(anchor_temperature),
        beta=beta,
        fuzziness_weight=fuzziness_weight,
        validation_interval=int(validation_interval),
        validation_start=validation_start,
        save_interval=int(save_interval),
        log_interval=max(1, int(log_interval)),
        parallel_mode=parallel_mode,
        gpu_ids=gpu_ids,
        mixed_precision=bool(mixed_precision),
        sync_batchnorm=bool(sync_batchnorm),
        linear_lr_scaling=bool(linear_lr_scaling),
        ddp_find_unused_parameters=bool(ddp_find_unused_parameters),
        resume=Path(resume) if resume is not None else None,
        checkpoint=Path(checkpoint) if checkpoint is not None else None,
        save_predictions=bool(save_predictions),
    )


def config_from_json_dict(values):
    """Rebuild a resolved configuration serialized by :meth:`to_json_dict`.

    This intentionally does not call ``build_config`` again: values such as a
    linearly-scaled learning rate have already been resolved before launch.
    Re-running the builder would apply those transforms twice.
    """
    payload = dict(values)
    payload["data_root"] = Path(payload["data_root"])
    payload["output"] = Path(payload["output"])
    payload["patch_size"] = tuple(payload.get("patch_size", (112, 112, 80)))
    payload["gpu_ids"] = tuple(payload.get("gpu_ids", (0,)))
    if payload.get("resume") is not None:
        payload["resume"] = Path(payload["resume"])
    if payload.get("checkpoint") is not None:
        payload["checkpoint"] = Path(payload["checkpoint"])

    # Backward-compatible defaults for launch files created before the studio
    # runtime fields were introduced.
    defaults = {
        "experiment_name": "",
        "log_interval": 10,
        "parallel_mode": "single",
        "gpu_ids": (0,),
        "mixed_precision": False,
        "sync_batchnorm": False,
        "linear_lr_scaling": False,
        "ddp_find_unused_parameters": False,
        "dnp_segmentation_weight": 0.5,
        "dnp_contrastive_weight": 0.05,
        "dnp_consistency": 1.0,
        "dnp_consistency_rampup": 40.0,
        "dnp_lds_weight": 1.0,
        "dnp_distribution_weight": 0.01,
        "dnp_np_supervised_weight": 0.1,
        "dnp_vat_magnitude": 10.0,
    }
    for key, value in defaults.items():
        payload.setdefault(key, value)
    return ExperimentConfig(**payload)

