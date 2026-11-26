"""End-user sidebar for the medical multimodal foundation-model workspace.

Engineering details (vLLM process layout, benchmark contracts, GraphRAG stores,
etc.) stay behind this UI. The user sees only model, inference, fine-tuning,
knowledge enhancement and compute-resource controls.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from foundation_model.data.jsonl_dataset import validate_jsonl_dataset
from foundation_model.jobs.manager import FoundationJobManager
from foundation_model.models.registry import get_foundation_model, list_foundation_models
from foundation_model.rag.index import GraphRAGIndexManager
from foundation_model.rag.embeddings import validate_local_model_dir
from foundation_model.rag.defaults import default_local_embedding_model
from foundation_model.rag.build_jobs import MedicalKBJobManager
from foundation_model.rag.local_backend import load_cached_knowledge_base
from foundation_model.rag.pipeline import HybridGraphRAGPipeline
from foundation_model.serving.vllm_manager import VLLMServiceManager
from studio.hardware import hardware_info
from studio.foundation_ui_state import is_foundation_training_running
from studio.ui.path_widgets import inline_directory_input, path_input





@st.fragment(run_every="2s")
def _render_training_status(manager: FoundationJobManager):
    status = manager.status()
    state = str(status.get("state", "idle"))
    if state == "idle":
        return
    progress = float(status.get("progress") or 0.0)
    st.progress(min(1.0, max(0.0, progress)))
    if status.get("loss") is not None:
        text = (
            f"{state} · Step {status.get('global_step', 0)}/{status.get('max_steps', '?')} · "
            f"Train loss {float(status['loss']):.4f}"
        )
        if status.get("eval_loss") is not None:
            text += f" · Val loss {float(status['eval_loss']):.4f}"
        st.caption(text)
    else:
        st.caption(f"状态：{state}")
    if status.get("best_eval_loss") is not None:
        st.caption(f"Best val loss: {float(status['best_eval_loss']):.4f}")
    if status.get("adapter_path"):
        st.caption(f"Best adapter: `{status['adapter_path']}`")
    if status.get("benchmark_samples"):
        st.caption(
            f"验证生成评估：{int(status['benchmark_samples'])} 条 · "
            f"Exact Match {float(status.get('benchmark_exact_match') or 0.0):.3f} · "
            f"Token F1 {float(status.get('benchmark_token_f1') or 0.0):.3f}"
        )
    if status.get("benchmark_output"):
        st.caption(f"评估结果：`{status['benchmark_output']}`")
    if status.get("benchmark_error"):
        st.warning(f"训练已完成，但生成式验证未完成：{status['benchmark_error']}")
    if state == "failed":
        st.error(status.get("error", "微调失败"))
    with st.expander("训练日志", expanded=False):
        st.code(manager.tail_log(120) or "暂无日志", language="text")

def _safe_index(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default


def _render_training_section(draft: dict, manager: FoundationJobManager, gpu_ids: list[int]):
    enabled = st.checkbox("启用模型微调", value=bool(draft.get("finetune_enabled", False)), key="fm_finetune_enabled")

    # Defaults are defined even while the section is disabled so the config contract
    # remains stable and later enabling it does not reset the user's settings.
    peft_strategy = str(draft.get("peft_strategy", "qlora"))
    distributed_strategy = str(draft.get("distributed_strategy", "ddp"))
    precision = str(draft.get("train_precision", "bf16"))
    batch = int(draft.get("per_device_train_batch_size", 1))
    grad_acc = int(draft.get("gradient_accumulation_steps", 8))
    epochs = float(draft.get("num_train_epochs", 1.0))
    lr = float(draft.get("learning_rate", 2e-5))
    rank = int(draft.get("lora_rank", 16))
    alpha = float(draft.get("lora_alpha", 16.0))
    dropout = float(draft.get("lora_dropout", 0.05))
    target_modules = str(draft.get("lora_target_modules", "all-linear"))
    gradient_checkpointing = bool(draft.get("gradient_checkpointing", True))
    resume = draft.get("resume_from_checkpoint") or None
    validation_dataset_path = draft.get("validation_dataset_path") or None
    validation_split = float(draft.get("validation_split", 0.1))
    auto_evaluate = bool(draft.get("auto_evaluate", True))
    evaluation_max_samples = int(draft.get("evaluation_max_samples", 20))
    evaluation_max_new_tokens = int(draft.get("evaluation_max_new_tokens", 128))

    if enabled:
        dataset_path = path_input(
            "训练数据",
            key="fm_dataset_path",
            value=str(draft.get("dataset_path", "examples/foundation_dataset_sample.jsonl")),
            mode="file",
            file_filter="JSONL (*.jsonl);;JSON (*.json);;All files (*)",
            help_text="多模态 SFT 数据，包含 image / question / answer。",
        )
        validation_dataset_path = path_input(
            "验证数据（可选）",
            key="fm_validation_dataset_path",
            value=str(validation_dataset_path or ""),
            mode="file",
            file_filter="JSONL (*.jsonl);;JSON (*.json);;All files (*)",
            help_text="留空时自动从训练数据中划分验证集。",
        ) or None
        if not validation_dataset_path:
            validation_split = float(
                st.number_input(
                    "自动验证集比例",
                    min_value=0.01,
                    max_value=0.50,
                    value=float(validation_split),
                    step=0.01,
                    key="fm_validation_split",
                )
            )

        if st.button("检查数据", use_container_width=True, key="fm_check_dataset"):
            try:
                report = validate_jsonl_dataset(dataset_path, check_images=True)
                st.success(f"训练数据：{report['samples']} 条，图像路径正常")
                first = report.get("first_sample") or {}
                if first:
                    st.caption(f"示例问题：{first.get('question', '')}")
                    st.caption(f"示例答案：{first.get('answer', '')}")
                if validation_dataset_path:
                    val_report = validate_jsonl_dataset(validation_dataset_path, check_images=True)
                    st.success(f"验证数据：{val_report['samples']} 条，图像路径正常")
                elif report["samples"] >= 2:
                    approx = max(1, int(round(report["samples"] * validation_split)))
                    st.caption(f"将自动划分约 {approx} 条作为验证集。")
            except Exception as exc:
                st.error(f"数据检查失败：{exc}")

        with st.expander("训练后验证", expanded=False):
            auto_evaluate = st.checkbox(
                "训练结束后自动生成回答并评估",
                value=auto_evaluate,
                key="fm_auto_evaluate",
                help="使用验证集和 best adapter 做生成式评估，不会再次加载基础模型。",
            )
            evaluation_max_samples = int(st.number_input(
                "最多评估样本数", min_value=1, value=evaluation_max_samples, step=1, key="fm_eval_max_samples"
            ))
            evaluation_max_new_tokens = int(st.number_input(
                "评估最大生成 tokens", min_value=16, max_value=2048, value=evaluation_max_new_tokens, step=16, key="fm_eval_max_tokens"
            ))

        experiment_name = st.text_input(
            "实验名称", value=str(draft.get("experiment_name", "medvlm_qlora")), key="fm_experiment_name"
        )
        peft_strategy = st.selectbox(
            "微调方式", ["qlora", "lora"],
            index=_safe_index(["qlora", "lora"], peft_strategy),
            format_func=lambda x: "QLoRA（推荐）" if x == "qlora" else "LoRA",
            key="fm_peft_strategy",
        )
        distributed_strategy = st.selectbox(
            "分布式策略", ["ddp", "deepspeed_zero2"],
            index=_safe_index(["ddp", "deepspeed_zero2"], distributed_strategy),
            format_func=lambda x: "Accelerate + DDP/NCCL" if x == "ddp" else "Accelerate + DeepSpeed ZeRO-2",
            key="fm_distributed_strategy",
        )
        precision = st.selectbox(
            "训练精度", ["bf16", "fp16"], index=_safe_index(["bf16", "fp16"], precision), key="fm_train_precision"
        )
        batch = int(st.number_input("每 GPU batch", min_value=1, value=batch, step=1, key="fm_batch"))
        grad_acc = int(st.number_input("Gradient accumulation", min_value=1, value=grad_acc, step=1, key="fm_grad_acc"))
        epochs = float(st.number_input("Epoch", min_value=0.1, value=epochs, step=0.5, key="fm_epochs"))
        lr = float(st.number_input("Learning rate", min_value=1e-7, value=lr, format="%.7f", key="fm_lr"))
        with st.expander("LoRA 参数", expanded=False):
            rank = int(st.number_input("LoRA rank", min_value=1, value=rank, step=1, key="fm_rank"))
            alpha = float(st.number_input("LoRA alpha", min_value=0.1, value=alpha, step=1.0, key="fm_alpha"))
            dropout = float(st.number_input("LoRA dropout", min_value=0.0, max_value=0.9, value=dropout, step=0.01, key="fm_dropout"))
            target_modules = st.text_input("Target modules", value=target_modules, key="fm_target_modules")
            gradient_checkpointing = st.checkbox("Gradient checkpointing", value=gradient_checkpointing, key="fm_grad_ckpt")
            resume = path_input(
                "Resume checkpoint",
                key="fm_resume",
                value=str(resume or ""),
                mode="directory",
                help_text="可选。留空表示新训练。",
            ) or None

        world = max(1, len(gpu_ids))
        st.caption(f"Global batch = {batch} × {world} GPU × {grad_acc} accumulation = {batch * world * grad_acc}")

        status = manager.status()
        running = is_foundation_training_running(status)
        b1, b2 = st.columns(2)
        if b1.button("开始微调", type="primary", disabled=bool(running), use_container_width=True, key="fm_start_train"):
            try:
                meta = manager.start({
                    **draft,
                    "experiment_name": experiment_name.strip(),
                    "dataset_path": dataset_path.strip(),
                    "validation_dataset_path": validation_dataset_path,
                    "validation_split": validation_split,
                    "auto_evaluate": auto_evaluate,
                    "evaluation_max_samples": evaluation_max_samples,
                    "evaluation_max_new_tokens": evaluation_max_new_tokens,
                    "gpu_ids": [int(i) for i in gpu_ids],
                    "distributed_strategy": distributed_strategy,
                    "peft_strategy": peft_strategy,
                    "train_precision": precision,
                    "per_device_train_batch_size": batch,
                    "gradient_accumulation_steps": grad_acc,
                    "num_train_epochs": epochs,
                    "learning_rate": lr,
                    "lora_rank": rank,
                    "lora_alpha": alpha,
                    "lora_dropout": dropout,
                    "lora_target_modules": target_modules,
                    "gradient_checkpointing": gradient_checkpointing,
                    "resume_from_checkpoint": resume,
                })
                st.success(f"微调已启动：{meta['job_id']}")
                st.rerun()
            except Exception as exc:
                st.error(f"启动失败：{exc}")
        if b2.button("停止", disabled=bool(not running), use_container_width=True, key="fm_stop_train"):
            st.write(manager.stop())
            st.rerun()

        _render_training_status(manager)
    else:
        dataset_path = str(draft.get("dataset_path", "examples/foundation_dataset_sample.jsonl"))
        experiment_name = str(draft.get("experiment_name", "medvlm_qlora"))
        validation_dataset_path = draft.get("validation_dataset_path") or None
        validation_split = float(draft.get("validation_split", 0.1))
        auto_evaluate = bool(draft.get("auto_evaluate", True))
        evaluation_max_samples = int(draft.get("evaluation_max_samples", 20))
        evaluation_max_new_tokens = int(draft.get("evaluation_max_new_tokens", 128))

    return {
        "finetune_enabled": bool(enabled),
        "dataset_path": dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "validation_split": validation_split,
        "auto_evaluate": auto_evaluate,
        "evaluation_max_samples": evaluation_max_samples,
        "evaluation_max_new_tokens": evaluation_max_new_tokens,
        "experiment_name": experiment_name.strip(),
        "peft_strategy": peft_strategy,
        "distributed_strategy": distributed_strategy,
        "train_precision": precision,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": grad_acc,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "lora_target_modules": target_modules,
        "gradient_checkpointing": gradient_checkpointing,
        "resume_from_checkpoint": resume,
    }



def _render_kb_build_summary(report: dict):
    st.success("✓ 医学知识库构建完成")
    c1, c2, c3 = st.columns(3)
    c1.metric("成功文档", int(report.get("documents", 0)))
    c2.metric("Snippets", int(report.get("chunks", 0)))
    c3.metric("医学概念", int(report.get("entities", 0)))
    c1, c2, c3 = st.columns(3)
    c1.metric("扫描型 PDF 跳过", int(report.get("scanned_pdf_files", 0)))
    c2.metric("解析失败", int(report.get("failed_files", 0)))
    c3.metric("图关系", int(report.get("relations", 0)))
    if int(report.get("pdf_pages_total", 0)):
        st.caption(
            f"PDF 页面：{int(report.get('pdf_pages_total', 0))} · "
            f"有效文本页：{int(report.get('pdf_pages_with_text', 0))} · "
            f"跳过页：{int(report.get('pdf_pages_skipped', 0))}"
        )
    st.caption(f"保存位置：`{report.get('index_dir', '')}`")

    warnings = list(report.get("warnings") or [])
    scanned = list(report.get("scanned_files") or [])
    failed = list(report.get("failed_file_details") or [])
    if warnings or scanned or failed:
        with st.expander("查看构建报告", expanded=False):
            if scanned:
                st.markdown("**扫描/图像型 PDF（未做 OCR，已跳过）**")
                for item in scanned[:200]:
                    st.caption(f"⚠ {item}")
            if failed:
                st.markdown("**解析失败**")
                for item in failed[:200]:
                    st.caption(f"✕ {item.get('source', '')} · {item.get('reason', '')}")
            if warnings and not (scanned or failed):
                for item in warnings[:200]:
                    st.caption(item)


def _resolve_kb_output_dir(parent_dir: str, kb_name: str) -> Path:
    parent = Path(parent_dir).expanduser().resolve()
    name = str(kb_name or "").strip()
    if not name:
        raise ValueError("请输入知识库名称。")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("知识库名称只能是单个文件夹名称，不能包含 /。")
    return (parent / name).resolve()


def _kb_required_files(index_dir: Path) -> list[str]:
    required = [
        GraphRAGIndexManager.MANIFEST,
        GraphRAGIndexManager.SNIPPETS,
        GraphRAGIndexManager.CHUNKS,
        GraphRAGIndexManager.EMBEDDINGS,
        GraphRAGIndexManager.GRAPH,
        GraphRAGIndexManager.BUILD_REPORT,
    ]
    return [name for name in required if not (index_dir / name).is_file()]


def _render_built_file_list(index_dir: Path) -> None:
    expected = [
        GraphRAGIndexManager.MANIFEST,
        GraphRAGIndexManager.SNIPPETS,
        GraphRAGIndexManager.CHUNKS,
        GraphRAGIndexManager.EMBEDDINGS,
        GraphRAGIndexManager.GRAPH,
        GraphRAGIndexManager.BUILD_REPORT,
    ]
    st.caption(f"实际保存目录：`{index_dir}`")
    with st.expander("查看已生成文件", expanded=True):
        for name in expected:
            path = index_dir / name
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                st.caption(f"✓ {name} · {size_mb:.2f} MB")
            else:
                st.caption(f"✕ {name} · 缺失")


@st.fragment(run_every="1s")
def _render_kb_build_status(job_manager: MedicalKBJobManager, expected_output_dir: str = ""):
    status = job_manager.status()
    state = str(status.get("state", "idle"))
    if state == "idle":
        return
    output_dir = str(status.get("output_dir") or "")
    if expected_output_dir and output_dir and str(Path(output_dir).resolve()) != str(Path(expected_output_dir).resolve()):
        st.caption(f"当前后台任务目标：`{output_dir}`")

    progress = float(status.get("progress") or 0.0)
    st.progress(max(0.0, min(1.0, progress)))
    message = str(status.get("message") or "正在构建医学知识库……")
    phase = str(status.get("phase") or "")
    if state in {"starting", "running"}:
        st.markdown(f"**{message}**")
        current = status.get("current")
        total = status.get("total")
        if current is not None and total:
            st.caption(f"阶段：{phase} · 进度：{current} / {total}")
        elif phase:
            st.caption(f"阶段：{phase}")
        return

    if state == "stopped":
        st.warning(message or "构建已停止。")
        return
    if state == "failed":
        st.error(f"构建失败：{status.get('error') or message}")
        log = job_manager.tail_log(80)
        if log:
            with st.expander("查看构建日志", expanded=False):
                st.code(log, language="text")
        return
    if state == "completed" and output_dir:
        manager = GraphRAGIndexManager(output_dir)
        validation = manager.validate()
        missing = _kb_required_files(Path(output_dir))
        if not validation.get("valid") or missing:
            details = validation.get("error") or "索引校验失败"
            if missing:
                details += "；缺失文件：" + ", ".join(missing)
            st.error(f"后台任务结束，但知识库校验失败：{details}")
            return
        st.success("医学知识库构建并落盘验证完成。")
        report = status.get("report") or manager.build_report()
        if report:
            _render_kb_build_summary(report)
        _render_built_file_list(Path(output_dir))
        key_suffix = str(status.get("job_id") or "latest")
        if st.button("使用此知识库", type="primary", use_container_width=True, key=f"fm_kb_use_completed_{key_suffix}"):
            st.session_state["fm_rag_active_kb"] = str(Path(output_dir).resolve())
            st.session_state["fm_rag_kb_mode"] = "home"
            load_cached_knowledge_base.cache_clear()
            st.rerun()


def _medical_kb_builder_panel(draft: dict, project_root):
    project_root = Path(project_root).resolve()
    default_source = project_root / "workspace" / "foundation" / "knowledge_docs"
    default_parent = project_root / "workspace" / "foundation"
    default_source.mkdir(parents=True, exist_ok=True)
    default_parent.mkdir(parents=True, exist_ok=True)
    job_manager = MedicalKBJobManager(project_root)

    st.caption("把用户提供的医学 PDF / TXT / MD / DOCX / JSON / JSONL 转换成可直接用于 GraphRAG 的完整知识库。扫描型 PDF 当前不做 OCR，会自动报告并跳过。")
    source_dir = inline_directory_input(
        "医学语料目录",
        key="fm_kb_builder_source",
        value=str(draft.get("knowledge_source_dir") or default_source),
        help_text="点击“选择目录”浏览远程服务器目录；会递归扫描其中支持的医学文件。",
        button_label="选择目录",
    ).strip()
    output_parent = inline_directory_input(
        "知识库父目录",
        key="fm_kb_builder_output_parent",
        value=str(default_parent),
        help_text="选择知识库文件夹要创建在哪个父目录下。程序会在这里新建一个以“知识库名称”命名的子目录。",
        button_label="选择目录",
        allow_create=True,
    ).strip()
    if "fm_kb_builder_name" not in st.session_state:
        st.session_state["fm_kb_builder_name"] = "medical_kb"
    kb_name = st.text_input(
        "知识库名称",
        key="fm_kb_builder_name",
        help="例如：lung_ct_kb。最终目录会是 <父目录>/<知识库名称>/。",
    ).strip()

    output_dir = None
    if output_parent and kb_name:
        try:
            output_dir = _resolve_kb_output_dir(output_parent, kb_name)
            st.caption(f"最终将保存到：`{output_dir}`")
        except Exception as exc:
            st.error(str(exc))

    with st.expander("高级设置", expanded=False):
        chunk_size = int(st.number_input(
            "Snippet size（字符）", min_value=256, max_value=4000,
            value=int(draft.get("rag_chunk_size", 1200)), step=100, key="fm_kb_builder_chunk_size",
        ))
        chunk_overlap = int(st.number_input(
            "Overlap（字符）", min_value=0, max_value=1000,
            value=int(draft.get("rag_chunk_overlap", 150)), step=25, key="fm_kb_builder_chunk_overlap",
        ))

        st.markdown("**Embedding 模型**")
        st.caption("固定使用轻量英文 BGE：BAAI/bge-small-en-v1.5（384 维）。建库阶段不再联网下载模型。")
        # Prefer the last user-selected directory; a local environment variable
        # can supply a default on unattended/server installs. Never assume a
        # developer-specific absolute path exists on other machines.
        widget_key = "fm_kb_builder_local_embedding"
        default_local_model = str(
            st.session_state.get(widget_key)
            or draft.get("rag_embedding_model")
            or default_local_embedding_model()
        ).strip()
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default_local_model

        embedding_backend = "local"
        embedding_model = inline_directory_input(
            "本地 Embedding 模型目录",
            key=widget_key,
            value=default_local_model,
            help_text="选择服务器上完整的 BAAI/bge-small-en-v1.5 模型目录。构建时只读本地文件，不访问 Hugging Face 或 ModelScope。",
            button_label="选择目录",
        ).strip()
        model_check = validate_local_model_dir(embedding_model)
        if model_check.get("valid"):
            st.success(f"Embedding 模型已就绪：{model_check['path']}")
        else:
            st.warning(
                "Embedding 模型尚未就绪。请先把 BAAI/bge-small-en-v1.5 完整模型上传到服务器，"
                "再选择该目录。开始构建按钮会保持禁用，不会再卡在下载阶段。\n\n"
                f"检查结果：{model_check.get('error', '未知错误')}"
            )

    status = job_manager.status()
    running = bool(status.get("process_alive") and status.get("state") in {"starting", "running"})
    can_build = bool(source_dir and output_dir is not None and model_check.get("valid"))

    c1, c2 = st.columns(2)
    if c1.button(
        "开始构建", type="primary", use_container_width=True,
        disabled=(not can_build) or running, key="fm_kb_start_build",
    ):
        try:
            assert output_dir is not None
            load_cached_knowledge_base.cache_clear()
            job_manager.start({
                "source_dir": source_dir,
                "output_dir": str(output_dir),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model or None,
                "embedding_device": "cpu",
                "batch_size": 16,
            })
            st.rerun()
        except Exception as exc:
            st.error(f"无法启动构建：{exc}")

    if c2.button(
        "停止构建", use_container_width=True, disabled=not running, key="fm_kb_stop_build",
    ):
        result = job_manager.stop()
        if not result.get("stopped"):
            st.warning(result.get("message", "没有正在运行的构建任务。"))
        st.rerun()

    _render_kb_build_status(job_manager, str(output_dir) if output_dir else "")

    # If a valid knowledge base already exists at the selected destination, show it
    # even after a server/UI restart (do not rely on session state).
    if output_dir is not None and not running:
        manager = GraphRAGIndexManager(output_dir)
        validation = manager.validate()
        latest_output = str(status.get("output_dir") or "")
        latest_completed_same = (
            status.get("state") == "completed" and latest_output and
            str(Path(latest_output).resolve()) == str(output_dir.resolve())
        )
        if validation.get("valid") and not latest_completed_same:
            st.info("目标目录中已经存在一个有效医学知识库。")
            report = manager.build_report()
            if report:
                _render_kb_build_summary(report)
            _render_built_file_list(output_dir)
            if st.button("使用此知识库", type="primary", use_container_width=True, key="fm_kb_use_existing"):
                st.session_state["fm_rag_active_kb"] = str(output_dir)
                st.session_state["fm_rag_kb_mode"] = "home"
                load_cached_knowledge_base.cache_clear()
                st.rerun()

def _medical_kb_import_panel(draft: dict, project_root):
    default_dir = str(draft.get("knowledge_base_dir") or "")
    index_dir = inline_directory_input(
        "已有知识库目录",
        key="fm_kb_import_dir",
        value=default_dir,
        help_text="点击“选择目录”浏览远程服务器，选择由本项目构建的完整 GraphRAG 知识库；导入不会重新生成 Embedding。",
        button_label="选择目录",
    ).strip()

    validation = None
    if index_dir:
        validation = GraphRAGIndexManager(index_dir).validate()
        if validation.get("valid"):
            st.success("✓ 有效医学知识库")
            c1, c2, c3 = st.columns(3)
            c1.metric("文档", int(validation.get("documents", 0)))
            c2.metric("Snippets", int(validation.get("chunks", 0)))
            c3.metric("医学概念", int(validation.get("entities", 0)))
            model_name = validation.get("embedding_model") or "内置轻量向量"
            st.caption(f"Embedding：{model_name} · 图关系：{int(validation.get('relations', 0))}")
        else:
            st.warning(validation.get("error", "当前目录不是有效的医学知识库。"))

    if st.button(
        "导入并使用", type="primary", use_container_width=True,
        disabled=not bool(validation and validation.get("valid")), key="fm_kb_import_use",
    ):
        st.session_state["fm_rag_active_kb"] = str(Path(index_dir).expanduser().resolve())
        st.session_state["fm_rag_kb_mode"] = "home"
        load_cached_knowledge_base.cache_clear()
        st.rerun()


def _reset_kb_inline_browsers():
    for key in (
        "fm_kb_builder_source__inline_browser_open",
        "fm_kb_builder_output_parent__inline_browser_open",
        "fm_kb_import_dir__inline_browser_open",
    ):
        st.session_state[key] = False


def _set_kb_mode(mode: str):
    st.session_state["fm_rag_kb_mode"] = mode
    _reset_kb_inline_browsers()


def _render_kb_mode_panel(draft: dict, project_root, mode: str):
    """Render KB build/import inline so Streamlit reruns cannot close it."""
    with st.container(border=True):
        if mode == "build":
            head1, head2 = st.columns([4, 1])
            head1.markdown("**构建医学语料库**")
            if head2.button("返回", key="fm_kb_builder_back", use_container_width=True):
                _set_kb_mode("home")
                st.rerun()
            _medical_kb_builder_panel(draft, project_root)
        elif mode == "import":
            head1, head2 = st.columns([4, 1])
            head1.markdown("**导入已有医学语料库**")
            if head2.button("返回", key="fm_kb_import_back", use_container_width=True):
                _set_kb_mode("home")
                st.rerun()
            _medical_kb_import_panel(draft, project_root)


def _render_rag_section(draft: dict, project_root):
    project_root = Path(project_root).resolve()
    mode = str(st.session_state.get("fm_rag_kb_mode", "home") or "home")

    if "fm_rag_active_kb" in st.session_state:
        active_dir = str(st.session_state.get("fm_rag_active_kb") or "").strip()
    else:
        active_dir = str(draft.get("knowledge_base_dir") or "").strip()
    active_manager = GraphRAGIndexManager(active_dir) if active_dir else None
    active_status = active_manager.status() if active_manager else {"ready": False}

    base_return = {
        "rag_enabled": False,
        "knowledge_source_dir": draft.get("knowledge_source_dir") or None,
        "knowledge_base_dir": None,
        "rag_embedding_backend": str(draft.get("rag_embedding_backend", "local")),
        "rag_embedding_model": str(draft.get("rag_embedding_model") or default_local_embedding_model()),
        "rag_top_k": int(draft.get("rag_top_k", 5)),
        "rag_graph_hops": int(draft.get("rag_graph_hops", 1)),
        "rag_chunk_size": int(draft.get("rag_chunk_size", 1200)),
        "rag_chunk_overlap": int(draft.get("rag_chunk_overlap", 150)),
        "rag_context_max_chars": int(draft.get("rag_context_max_chars", 9000)),
        "rag_visual_max_new_tokens": int(draft.get("rag_visual_max_new_tokens", 192)),
    }

    # KB builder/importer is intentionally rendered inline. Directory navigation
    # calls st.rerun(), and the persistent mode keeps this panel alive across
    # every rerun. This avoids the disappearing-dialog failure mode.
    if mode in {"build", "import"}:
        _render_kb_mode_panel(draft, project_root, mode)
        if not active_status.get("ready"):
            return base_return
        st.divider()

    if not active_status.get("ready"):
        st.caption("先建立或导入一个医学语料库，之后才显示 GraphRAG 参数。")
        c1, c2 = st.columns(2)
        if c1.button("构建医学语料库", type="primary", use_container_width=True, key="fm_rag_open_builder"):
            _set_kb_mode("build")
            st.rerun()
        if c2.button("导入已有语料库", use_container_width=True, key="fm_rag_open_import"):
            _set_kb_mode("import")
            st.rerun()
        return base_return

    st.success("● 医学语料库已加载")
    model_name = active_status.get("embedding_model") or "内置轻量向量"
    st.caption(
        f"{int(active_status.get('documents', 0))} 文档 · {int(active_status.get('chunks', 0))} snippets · "
        f"{int(active_status.get('entities', 0))} concepts · {int(active_status.get('relations', 0))} edges"
    )
    st.caption(f"Embedding：{model_name}")
    st.caption(f"知识库：`{active_dir}`")

    c1, c2 = st.columns(2)
    if c1.button("新建语料库", use_container_width=True, key="fm_rag_new_kb"):
        _set_kb_mode("build")
        st.rerun()
    if c2.button("更换 / 导入", use_container_width=True, key="fm_rag_change_kb"):
        _set_kb_mode("import")
        st.rerun()

    enabled = st.checkbox(
        "启用医学知识增强",
        value=bool(draft.get("rag_enabled", True)),
        key="fm_rag_enabled",
        help="图像线索 + 用户问题共同形成检索 Query，再用原图 + 原问题 + 医学证据生成最终回答。",
    )

    st.markdown("**GraphRAG 设置**")
    top_k = int(st.number_input(
        "检索 Top-K", min_value=1, max_value=20,
        value=int(draft.get("rag_top_k", 5)), step=1, key="fm_rag_top_k",
    ))
    graph_hops = int(st.selectbox(
        "图扩展 hops", [0, 1, 2],
        index=_safe_index([0, 1, 2], int(draft.get("rag_graph_hops", 1))),
        key="fm_rag_graph_hops", help="一般使用 1 hop；2 hops 更容易引入无关概念。",
    ))
    context_max_chars = int(st.number_input(
        "最大证据字符数", min_value=1000, max_value=30000,
        value=int(draft.get("rag_context_max_chars", 9000)), step=500, key="fm_rag_context_max_chars",
    ))
    visual_max_new_tokens = int(st.number_input(
        "图像线索最大 tokens", min_value=64, max_value=512,
        value=int(draft.get("rag_visual_max_new_tokens", 192)), step=32, key="fm_rag_visual_tokens",
        help="第一次 VLM 只提取检索线索，不生成最终回答。",
    ))

    with st.expander("知识库构建信息", expanded=False):
        st.caption(
            f"Snippet size：{int(active_status.get('chunk_size', 1200))} · "
            f"Overlap：{int(active_status.get('chunk_overlap', 150))} · "
            f"格式版本：v{int(active_status.get('version', 1))}"
        )
        report = active_manager.build_report()
        if report:
            _render_kb_build_summary(report)
        elif not active_status.get("portable_snippets"):
            st.caption("这是旧版知识库格式；仍可正常检索，但没有独立 build_report/snippets.jsonl。")

    with st.expander("检索测试", expanded=False):
        test_query = st.text_input("测试问题", key="fm_rag_test_query", placeholder="例如：B细胞淋巴瘤的典型病理特征")
        if st.button("检索", use_container_width=True, key="fm_rag_test_button", disabled=not bool(test_query.strip())):
            try:
                kb = load_cached_knowledge_base(active_dir, active_manager.fingerprint())
                context = HybridGraphRAGPipeline(kb).retrieve(
                    test_query, top_k=top_k, graph_hops=graph_hops, max_context_chars=context_max_chars,
                )
                if not context.sources:
                    st.caption("没有检索到证据。")
                for source in context.sources:
                    meta = source.get("metadata") or {}
                    location = f" · p.{meta['page']}" if meta.get("page") else ""
                    st.markdown(
                        f"**[{source['index']}] {source['source']}{location}**  "
                        f"score={float(source['score']):.3f} · {source['retrieval_path']}"
                    )
                    st.caption(source.get("text", "")[:500])
            except Exception as exc:
                st.error(f"检索失败：{exc}")

    if st.button("卸载当前语料库", use_container_width=True, key="fm_rag_unload_kb"):
        st.session_state["fm_rag_active_kb"] = ""
        st.session_state["fm_rag_kb_mode"] = "home"
        load_cached_knowledge_base.cache_clear()
        st.rerun()

    return {
        "rag_enabled": bool(enabled),
        "knowledge_source_dir": active_status.get("source_dir") or draft.get("knowledge_source_dir") or None,
        "knowledge_base_dir": active_dir,
        "rag_embedding_backend": str(active_status.get("embedding_backend") or draft.get("rag_embedding_backend", "local")),
        "rag_embedding_model": str(active_status.get("embedding_model") or draft.get("rag_embedding_model") or default_local_embedding_model()),
        "rag_top_k": top_k,
        "rag_graph_hops": graph_hops,
        "rag_chunk_size": int(active_status.get("chunk_size", draft.get("rag_chunk_size", 1200))),
        "rag_chunk_overlap": int(active_status.get("chunk_overlap", draft.get("rag_chunk_overlap", 150))),
        "rag_context_max_chars": context_max_chars,
        "rag_visual_max_new_tokens": visual_max_new_tokens,
    }


def render_foundation_sidebar(store, draft, project_root):
    hw = hardware_info()
    train_manager = FoundationJobManager(project_root)
    service_manager = VLLMServiceManager(project_root)
    st.sidebar.header("医疗多模态助手")

    with st.sidebar.expander("① 模型", expanded=True):
        specs = list_foundation_models()
        names = {s.display_name: s.key for s in specs}
        current_key = draft.get("model_key", "medgemma4b")
        current_name = next((s.display_name for s in specs if s.key == current_key), specs[0].display_name)
        selected_name = st.selectbox("模型", list(names), index=_safe_index(list(names), current_name), key="fm_model_key")
        model_key = names[selected_name]
        spec = get_foundation_model(model_key)
        default_id = spec.default_model_id if model_key != "custom_local_vlm" else str(draft.get("base_model", ""))
        if st.session_state.get("fm_last_model_key") != model_key:
            st.session_state["fm_last_model_key"] = model_key
            if model_key != "custom_local_vlm":
                st.session_state["fm_base_model"] = default_id
        base_model = path_input(
            "模型 / 本地路径",
            key="fm_base_model",
            value=str(st.session_state.get("fm_base_model", draft.get("base_model") or default_id)),
            mode="directory",
            help_text="Hugging Face model id 可直接填写；本地模型点击 Open Dir 从服务器选择。",
            editable=True,
        ).strip()
        st.caption(spec.description)

    with st.sidebar.expander("② 推理", expanded=True):
        inference_backend = st.selectbox(
            "推理方式",
            ["vllm", "transformers"],
            index=_safe_index(["vllm", "transformers"], draft.get("inference_backend", "vllm")),
            format_func=lambda x: "高性能推理" if x == "vllm" else "兼容模式",
            key="fm_inference_backend",
            help="高性能推理底层使用 vLLM；兼容模式使用隔离的 Transformers 子进程。",
        )
        adapter_path = path_input(
            "Adapter",
            key="fm_adapter_path",
            value=str(draft.get("adapter_path") or ""),
            mode="directory",
            help_text="可选。选择 LoRA/QLoRA 输出目录；留空使用基础模型。",
        ) or None

        serving_strategy = str(draft.get("serving_strategy", "data_parallel"))
        serving_precision = str(draft.get("serving_precision", "bf16"))
        inference_precision = str(draft.get("inference_precision", "bf16"))
        inference_quantization = str(draft.get("inference_quantization", "4bit_nf4"))
        if inference_backend == "vllm":
            serving_precision = st.selectbox(
                "推理精度", ["bf16", "int4"],
                index=_safe_index(["bf16", "int4"], serving_precision),
                format_func=lambda x: x.upper(), key="fm_serving_precision",
            )
            serving_strategy = st.selectbox(
                "多 GPU 推理策略", ["data_parallel", "tensor_parallel"],
                index=_safe_index(["data_parallel", "tensor_parallel"], serving_strategy),
                format_func=lambda x: "多副本并行" if x == "data_parallel" else "张量并行",
                key="fm_serving_strategy",
            )
            state = service_manager.status()
            running_service = bool(state.get("state") == "running")
            c1, c2 = st.columns(2)
            if c1.button("加载模型", type="primary", disabled=running_service, use_container_width=True, key="fm_load_service"):
                try:
                    from studio.foundation_page import build_serving_config
                    cfg = build_serving_config({
                        **draft,
                        "base_model": base_model,
                        "adapter_path": adapter_path,
                        "serving_strategy": serving_strategy,
                        "serving_precision": serving_precision,
                        "gpu_ids": [int(i) for i in st.session_state.get("fm_gpus", draft.get("gpu_ids", []))],
                    })
                    service_manager.start(cfg)
                    st.rerun()
                except Exception as exc:
                    st.error(f"模型加载失败：{exc}")
            if c2.button("卸载", disabled=not running_service, use_container_width=True, key="fm_unload_service"):
                st.write(service_manager.stop())
                st.rerun()
            state = service_manager.status()
            if state.get("healthy"):
                st.success("● 已就绪")
            elif state.get("state") == "running":
                st.warning("● 正在启动")
            else:
                st.caption("○ 未加载")
        else:
            inference_precision = st.selectbox(
                "推理精度", ["bf16", "fp16"],
                index=_safe_index(["bf16", "fp16"], inference_precision),
                format_func=lambda x: x.upper(), key="fm_inference_precision",
            )
            inference_quantization = st.selectbox(
                "VLM 推理量化", ["4bit_nf4", "none"],
                index=_safe_index(["4bit_nf4", "none"], inference_quantization),
                format_func=lambda x: "4-bit NF4（推荐）" if x == "4bit_nf4" else "不量化",
                key="fm_inference_quantization",
                help="只量化 VLM。Embedding/向量库/GraphRAG 不做 4-bit 量化。",
            )
            st.caption("兼容模式按请求启动隔离进程：提交问题时自动加载模型；启用 GraphRAG 时，同一份量化 VLM 会在单个子进程内串行完成“图像线索提取 → 最终回答”，结束后释放 GPU。")

    with st.sidebar.expander("③ 微调", expanded=False):
        # GPU IDs are chosen below; use current draft IDs for this render. The next
        # rerun persists the new selection before a training job can be launched.
        draft_gpu_ids = [int(i) for i in st.session_state.get("fm_gpus", draft.get("gpu_ids", []))]
        training_values = _render_training_section(draft, train_manager, draft_gpu_ids)

    # Keep the whole knowledge-enhancement section open while the user is
    # building/importing a KB. Directory navigation uses st.rerun(), and an
    # expander recreated with expanded=False would otherwise look as if the
    # builder "disappeared" after every click.
    rag_mode = str(st.session_state.get("fm_rag_kb_mode", "home") or "home")
    rag_browser_open = any(
        bool(st.session_state.get(key, False))
        for key in (
            "fm_kb_builder_source__inline_browser_open",
            "fm_kb_builder_output_parent__inline_browser_open",
            "fm_kb_import_dir__inline_browser_open",
        )
    )
    with st.sidebar.expander(
        "④ 医学知识增强",
        expanded=(rag_mode in {"build", "import"} or rag_browser_open),
    ):
        rag_values = _render_rag_section(draft, project_root)

    with st.sidebar.expander("⑤ 计算资源", expanded=True):
        if hw["cuda_available"]:
            options = [g["id"] for g in hw["gpus"]]
            default = [int(i) for i in draft.get("gpu_ids", options[: min(2, len(options))]) if int(i) in options]
            if not default and options:
                default = [options[0]]
            gpu_ids = st.multiselect(
                "GPU",
                options,
                default=default,
                format_func=lambda i: f"GPU {i}: {next(g['name'] for g in hw['gpus'] if g['id'] == i)}",
                key="fm_gpus",
            )
            if not gpu_ids:
                st.warning("至少选择一张 GPU。")
            else:
                st.caption(f"已选择 {len(gpu_ids)} 张 GPU · World size = {len(gpu_ids)}")
        else:
            gpu_ids = []
            st.error("未检测到 CUDA。")

    with st.sidebar.expander("⑥ 高级设置", expanded=False):
        seed = int(st.number_input("Random seed", min_value=0, value=int(draft.get("seed", 1337)), step=1, key="fm_seed"))
        trust_remote_code = st.checkbox("Trust remote code", value=bool(draft.get("trust_remote_code", False)), key="fm_trust_remote_code")
        max_new_tokens = int(st.number_input("Max new tokens", min_value=16, max_value=4096, value=int(draft.get("max_new_tokens", 256)), step=16, key="fm_max_new_tokens"))
        temperature = float(st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(draft.get("temperature", 0.0)), step=0.1, key="fm_temperature"))
        serving_host = st.text_input("服务地址", value=str(draft.get("serving_host", "127.0.0.1")), key="fm_host")
        serving_port = int(st.number_input("Port", min_value=1, max_value=65535, value=int(draft.get("serving_port", 8010)), step=1, key="fm_port"))
        mem_util = float(st.slider("GPU memory utilization", 0.50, 0.95, float(draft.get("gpu_memory_utilization", 0.88)), 0.01, key="fm_mem_util"))
        if st.button("清空对话", use_container_width=True, key="fm_clear_chat"):
            st.session_state.pop("fm_chat_messages", None)
            st.session_state.pop("fm_chat_image_path", None)
            st.rerun()

    values = {
        **draft,
        **training_values,
        "model_key": model_key,
        "base_model": base_model,
        "gpu_ids": [int(i) for i in gpu_ids],
        "seed": seed,
        "trust_remote_code": bool(trust_remote_code),
        "adapter_path": adapter_path,
        "inference_backend": inference_backend,
        "inference_precision": inference_precision,
        "inference_quantization": inference_quantization,
        "serving_strategy": serving_strategy,
        "serving_precision": serving_precision,
        "serving_host": serving_host,
        "serving_port": serving_port,
        "gpu_memory_utilization": mem_util,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        **rag_values,
    }
    return store.write(values), hw
