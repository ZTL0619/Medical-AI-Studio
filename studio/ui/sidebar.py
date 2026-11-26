"""Modular Streamlit sidebar for 3-D segmentation experiment configuration."""

import streamlit as st

from ssnet_final.models.registry import get_model_spec, list_model_specs
from studio.hardware import hardware_info
from studio.ui.path_widgets import path_input


SIDEBAR_WIDGET_KEYS = (
    "ui_experiment_name", "ui_data_root", "ui_model", "ui_feature", "ui_semi",
    "ui_labeled_percent", "ui_custom_hparams", "ui_batch", "ui_labeled_batch",
    "ui_lr", "ui_iters", "ui_workers", "ui_seed", "ui_validation_interval",
    "ui_save_interval", "ui_log_interval", "ui_parallel", "ui_single_gpu",
    "ui_ddp_gpu_ids", "ui_amp", "ui_sync_bn", "ui_lr_scale", "ui_resume",
)


def reset_sidebar_widget_state():
    for key in SIDEBAR_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _model_section(draft):
    specs = list_model_specs()
    label_to_key = {spec.display_name: spec.key for spec in specs}
    current_key = draft.get("model", "vnet")
    current_label = next((s.display_name for s in specs if s.key == current_key), specs[0].display_name)
    model_label = st.selectbox(
        "模型", list(label_to_key), index=list(label_to_key).index(current_label), key="ui_model"
    )
    model = label_to_key[model_label]
    spec = get_model_spec(model)
    st.caption(spec.description)

    feature = "none"
    semi = "none"
    if model == "vnet":
        feature_labels = {"None": "none", "FSL": "fsl"}
        feature_default = "FSL" if draft.get("feature_learning") == "fsl" else "None"
        feature = feature_labels[st.selectbox(
            "Feature Learning", list(feature_labels),
            index=list(feature_labels).index(feature_default), key="ui_feature"
        )]
        semi_labels = {"None": "none", "Traditional SSL": "traditional", "DNP": "dnp"}
        reverse = {value: key for key, value in semi_labels.items()}
        semi_default = reverse.get(draft.get("semi_supervision", "none"), "None")
        semi = semi_labels[st.selectbox(
            "Semi-supervision", list(semi_labels),
            index=list(semi_labels).index(semi_default), key="ui_semi"
        )]
    else:
        st.info("CAT-Net 3D 当前使用监督训练。")

    labeled_options = [5, 10] if (feature == "fsl" or semi != "none") else [5, 10, 100]
    current_pct = int(draft.get("labeled_percent", 10))
    if current_pct not in labeled_options:
        current_pct = labeled_options[-1]
    if st.session_state.get("ui_labeled_percent") not in labeled_options:
        st.session_state.pop("ui_labeled_percent", None)
    labeled_percent = st.selectbox(
        "Labeled data", labeled_options, index=labeled_options.index(current_pct),
        format_func=lambda x: f"{x}%", key="ui_labeled_percent"
    )
    return model, feature, semi, int(labeled_percent), spec


def _training_section(draft, model, semi, labeled_percent):
    custom = st.checkbox(
        "自定义训练超参数",
        value=any(draft.get(k) is not None for k in ["batch_size", "learning_rate", "max_iterations"]),
        key="ui_custom_hparams",
    )
    if model == "catnet3d":
        default_batch, default_labeled, default_lr, default_iter = 1, 1, 1e-4, 40000
    else:
        default_batch = 4 if semi != "none" else 2
        default_labeled = 2 if semi != "none" else default_batch
        default_lr = 0.01
        default_iter = 15000 if semi == "dnp" and labeled_percent == 5 else 10000

    if custom:
        batch_size = int(st.number_input(
            "每 GPU batch size", min_value=1,
            value=int(draft.get("batch_size") or default_batch), step=1, key="ui_batch"
        ))
        if semi != "none":
            labeled_batch_size = int(st.number_input(
                "每 GPU labeled batch", min_value=1, max_value=max(1, batch_size - 1),
                value=min(int(draft.get("labeled_batch_size") or default_labeled), max(1, batch_size - 1)),
                step=1, key="ui_labeled_batch"
            ))
        else:
            labeled_batch_size = batch_size
        learning_rate = float(st.number_input(
            "Learning rate", min_value=1e-7,
            value=float(draft.get("learning_rate") or default_lr), format="%.6f", key="ui_lr"
        ))
        max_iterations = int(st.number_input(
            "Max iterations", min_value=1,
            value=int(draft.get("max_iterations") or default_iter), step=100, key="ui_iters"
        ))
    else:
        batch_size = labeled_batch_size = learning_rate = max_iterations = None

    validation_interval = int(st.number_input(
        "Validation interval", min_value=1,
        value=int(draft.get("validation_interval", 200)), step=50, key="ui_validation_interval"
    ))
    save_interval = int(st.number_input(
        "Last checkpoint interval", min_value=1,
        value=int(draft.get("save_interval", 1000)), step=100, key="ui_save_interval"
    ))
    log_interval = int(st.number_input(
        "Log interval", min_value=1,
        value=int(draft.get("log_interval", 10)), step=1, key="ui_log_interval"
    ))
    return batch_size, labeled_batch_size, learning_rate, max_iterations, validation_interval, save_interval, log_interval


def _acceleration_section(draft, hw):
    if hw["cuda_available"]:
        gpu_options = [gpu["id"] for gpu in hw["gpus"]]
        parallel_choices = ["single"] + (["ddp"] if len(gpu_options) >= 2 else [])
        default_parallel = draft.get("parallel_mode", "single")
        if default_parallel not in parallel_choices:
            default_parallel = "single"
        parallel_mode = st.radio(
            "训练模式", parallel_choices, index=parallel_choices.index(default_parallel),
            format_func=lambda x: "单 GPU" if x == "single" else "DDP 多 GPU",
            key="ui_parallel", horizontal=True,
        )
        previous = st.session_state.get("_ui_last_parallel_mode")
        if previous is not None and previous != parallel_mode:
            st.session_state.pop("ui_single_gpu", None)
            st.session_state.pop("ui_ddp_gpu_ids", None)
        st.session_state["_ui_last_parallel_mode"] = parallel_mode

        draft_gpus = [g for g in draft.get("gpu_ids", [0]) if g in gpu_options]
        if parallel_mode == "single":
            default_gpu = draft_gpus[0] if draft_gpus else gpu_options[0]
            single_gpu = st.selectbox(
                "GPU", gpu_options, index=gpu_options.index(default_gpu),
                format_func=lambda i: f"GPU {i}: {hw['gpus'][i]['name']}", key="ui_single_gpu"
            )
            gpu_ids = [int(single_gpu)]
        else:
            default_gpus = draft_gpus if len(draft_gpus) >= 2 else gpu_options[:2]
            gpu_ids = st.multiselect(
                "GPU", gpu_options, default=default_gpus,
                format_func=lambda i: f"GPU {i}: {hw['gpus'][i]['name']}", key="ui_ddp_gpu_ids"
            )
            if len(gpu_ids) < 2:
                st.warning("DDP 多 GPU 至少选择两张 GPU。")
        device = "cuda"
    else:
        st.warning("未检测到 CUDA，将使用 CPU。")
        parallel_mode, gpu_ids, device = "single", [0], "cpu"

    mixed_precision = st.checkbox("Mixed Precision (AMP)", value=bool(draft.get("mixed_precision", False)), key="ui_amp")
    sync_batchnorm = st.checkbox(
        "SyncBatchNorm", value=bool(draft.get("sync_batchnorm", False)),
        disabled=parallel_mode != "ddp", key="ui_sync_bn"
    )
    linear_lr_scaling = st.checkbox(
        "按 GPU 数线性缩放学习率", value=bool(draft.get("linear_lr_scaling", False)),
        disabled=parallel_mode != "ddp", key="ui_lr_scale"
    )
    return device, parallel_mode, gpu_ids, mixed_precision, sync_batchnorm, linear_lr_scaling


def render_sidebar(config_store, job_manager, draft):
    hw = hardware_info()
    st.sidebar.header("3D 分割配置")

    with st.sidebar.expander("① 数据与实验", expanded=True):
        experiment_name = st.text_input(
            "实验名称", value=str(draft.get("experiment_name", "")),
            placeholder="可选，例如 dnp_5pct_repro", key="ui_experiment_name"
        )
        data_root = path_input(
            "数据目录",
            key="ui_data_root",
            value=str(draft.get("data_root") or ""),
            mode="directory",
            help_text="UA-MT/LA 数据根目录；训练、验证和推理统一读取该路径。鼠标悬停可查看完整路径。",
        )

    with st.sidebar.expander("② 模型与训练策略", expanded=True):
        model, feature, semi, labeled_percent, spec = _model_section(draft)

    with st.sidebar.expander("③ 训练超参数", expanded=True):
        (
            batch_size, labeled_batch_size, learning_rate, max_iterations,
            validation_interval, save_interval, log_interval,
        ) = _training_section(draft, model, semi, labeled_percent)

    with st.sidebar.expander("④ 加速与设备", expanded=True):
        device, parallel_mode, gpu_ids, mixed_precision, sync_batchnorm, linear_lr_scaling = _acceleration_section(draft, hw)

    with st.sidebar.expander("⑤ 运行与复现", expanded=False):
        seed = int(st.number_input("Random seed", min_value=0, value=int(draft.get("seed", 1337)), step=1, key="ui_seed"))
        workers = int(st.number_input("DataLoader workers", min_value=0, value=int(draft.get("workers", 4)), step=1, key="ui_workers"))
        resume = path_input(
            "Resume checkpoint",
            key="ui_resume",
            value=str(draft.get("resume") or ""),
            mode="file",
            file_filter="Checkpoint (*.pt *.pth *.ckpt *.tar);;All files (*)",
            help_text="可选。点击 Open Files 从服务器选择 checkpoint；留空表示新训练。",
        ) or None
        st.caption("每个训练任务会自动保存 resolved config、环境版本和 reproducibility.json。")

    values = {
        **draft,
        "dataset": "LA",
        "experiment_name": experiment_name.strip(),
        "data_root": data_root,
        "model": model,
        "feature_learning": feature,
        "semi_supervision": semi,
        "labeled_percent": labeled_percent,
        "device": device,
        "parallel_mode": parallel_mode,
        "gpu_ids": [int(v) for v in gpu_ids],
        "mixed_precision": mixed_precision,
        "sync_batchnorm": sync_batchnorm if parallel_mode == "ddp" else False,
        "linear_lr_scaling": linear_lr_scaling if parallel_mode == "ddp" else False,
        "batch_size": batch_size,
        "labeled_batch_size": labeled_batch_size,
        "learning_rate": learning_rate,
        "max_iterations": max_iterations,
        "workers": workers,
        "seed": seed,
        "validation_interval": validation_interval,
        "save_interval": save_interval,
        "log_interval": log_interval,
        "resume": resume,
    }
    st.session_state.draft = config_store.write(values)

    try:
        _, resolved = job_manager.create_config(values, job_id="preview")
        st.sidebar.markdown("#### 配置摘要")
        st.sidebar.caption(
            f"{spec.display_name} · {resolved.optimizer.upper()} · "
            f"global batch={resolved.global_batch_size} · LR={resolved.learning_rate:g}"
        )
    except Exception as exc:
        st.sidebar.error(str(exc))
    return values, hw
