import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage.measure import marching_cubes

from studio.config_store import ConfigStore
from studio.dataset_tools import inspect_dataset, list_cases, load_case, normalize_slice, overlay_slice
from studio.inference import run_case_inference
from studio.job_manager import JobManager
from studio.foundation_config_store import FoundationConfigStore
from studio.foundation_page import render_foundation_workspace
from studio.ui.foundation_sidebar import render_foundation_sidebar
from studio.ui import render_sidebar, reset_sidebar_widget_state
from studio.ui.path_widgets import path_input
from ssnet_final.models.registry import get_model_spec


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "workspace"
CONFIG_STORE = ConfigStore(WORKSPACE / "current_config.json")
JOB_MANAGER = JobManager(PROJECT_ROOT)
FOUNDATION_CONFIG_STORE = FoundationConfigStore(WORKSPACE / "foundation_config.json")

st.set_page_config(
    page_title="Medical Image Segmentation Studio",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load_case(root, case):
    return load_case(root, case)


def _mesh_trace(volume, name, color):
    """Create a compact surface mesh for a binary 3-D segmentation."""
    foreground = (np.asarray(volume) > 0).astype(np.uint8)
    if not np.any(foreground):
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name=name)

    padded = np.pad(foreground, 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        step_size=2,
        allow_degenerate=False,
    )
    vertices -= 1.0
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=0.88,
        flatshading=False,
        hoverinfo="skip",
        showscale=False,
    )


@st.cache_data(show_spinner=False)
def linked_segmentation_3d_html(label, prediction):
    """Two independent Plotly 3-D scenes whose cameras are synchronized."""
    label = np.asarray(label)
    prediction = np.asarray(prediction)
    shape = np.maximum(np.asarray(label.shape), np.asarray(prediction.shape))

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Ground Truth", "Prediction"),
        horizontal_spacing=0.04,
    )
    figure.add_trace(_mesh_trace(label, "Ground Truth", "#4C78A8"), row=1, col=1)
    figure.add_trace(_mesh_trace(prediction, "Prediction", "#F58518"), row=1, col=2)

    initial_camera = dict(eye=dict(x=1.45, y=1.45, z=1.25), up=dict(x=0, y=0, z=1))
    axis = dict(showbackground=False, showspikes=False, zeroline=False)
    scene_common = dict(
        xaxis={**axis, "title": "X", "range": [0, float(shape[0] - 1)]},
        yaxis={**axis, "title": "Y", "range": [0, float(shape[1] - 1)]},
        zaxis={**axis, "title": "Z", "range": [0, float(shape[2] - 1)]},
        aspectmode="data",
        camera=initial_camera,
    )
    figure.update_layout(
        scene=scene_common,
        scene2=scene_common,
        height=540,
        margin=dict(l=0, r=0, t=45, b=0),
        showlegend=False,
        uirevision="linked-gt-pred",
    )

    post_script = r'''
var gd = document.getElementById('{plot_id}');
var syncingCamera = false;
gd.on('plotly_relayout', function(evt) {
    if (syncingCamera || !evt) return;
    var keys = Object.keys(evt);
    var scene1Changed = keys.some(function(k) {
        return k === 'scene.camera' || k.indexOf('scene.camera.') === 0;
    });
    var scene2Changed = keys.some(function(k) {
        return k === 'scene2.camera' || k.indexOf('scene2.camera.') === 0;
    });
    if (!scene1Changed && !scene2Changed) return;

    var source = scene1Changed ? 'scene' : 'scene2';
    var target = scene1Changed ? 'scene2' : 'scene';
    if (!gd.layout[source] || !gd.layout[source].camera) return;
    var camera = JSON.parse(JSON.stringify(gd.layout[source].camera));

    syncingCamera = true;
    var update = {};
    update[target + '.camera'] = camera;
    Promise.resolve(Plotly.relayout(gd, update)).then(function() {
        syncingCamera = false;
    }).catch(function() {
        syncingCamera = false;
    });
});
'''
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        post_script=post_script,
    )


def render_linked_segmentation_3d(label, prediction):
    html = linked_segmentation_3d_html(label, prediction)
    components.html(html, height=570, scrolling=False)



@st.cache_data(show_spinner=False)
def mri_gt_3d_figure(image, label, max_axis_points=56):
    """Interactive 3-D MRI volume preview with the GT surface overlaid.

    The MRI is downsampled only for rendering so that the data/preview page stays
    responsive; the segmentation GT surface remains in the original voxel
    coordinate system.
    """
    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label)
    if image.ndim != 3:
        raise ValueError(f"3D MRI preview expects a 3-D volume, got {image.shape}")

    # Keep the volume rendering compact enough for the browser while preserving
    # the original voxel coordinate system for axes and GT.
    stride = max(1, int(np.ceil(max(image.shape) / float(max_axis_points))))
    volume = image[::stride, ::stride, ::stride]
    finite = volume[np.isfinite(volume)]
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
        else:
            volume = np.zeros_like(volume, dtype=np.float32)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    xs = np.arange(0, image.shape[0], stride, dtype=np.float32)[: volume.shape[0]]
    ys = np.arange(0, image.shape[1], stride, dtype=np.float32)[: volume.shape[1]]
    zs = np.arange(0, image.shape[2], stride, dtype=np.float32)[: volume.shape[2]]
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")

    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=xg.ravel(),
            y=yg.ravel(),
            z=zg.ravel(),
            value=volume.ravel(),
            isomin=0.08,
            isomax=1.0,
            opacity=0.08,
            surface_count=12,
            colorscale="Gray",
            showscale=False,
            name="MRI",
            hoverinfo="skip",
        )
    )
    if np.any(label > 0):
        gt = _mesh_trace(label, "Ground Truth", "#4C78A8")
        gt.opacity = 0.50
        fig.add_trace(gt)

    axis = dict(showbackground=False, showspikes=False, zeroline=False)
    fig.update_layout(
        scene=dict(
            xaxis={**axis, "title": "X", "range": [0, float(image.shape[0] - 1)]},
            yaxis={**axis, "title": "Y", "range": [0, float(image.shape[1] - 1)]},
            zaxis={**axis, "title": "Z", "range": [0, float(image.shape[2] - 1)]},
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.25), up=dict(x=0, y=0, z=1)),
        ),
        height=610,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        uirevision=f"mri-preview-{image.shape}",
    )
    return fig

def init_state():
    if "draft" not in st.session_state:
        st.session_state.draft = CONFIG_STORE.read()
    if "dataset_report" not in st.session_state:
        st.session_state.dataset_report = None
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None
    if st.session_state.get("reload_widgets"):
        st.session_state.draft = CONFIG_STORE.read()
        reset_sidebar_widget_state()
        st.session_state.reload_widgets = False


def render_workspace(draft):
    st.subheader("数据与 3D MRI 预览")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("检查数据集", use_container_width=True):
            with st.spinner("正在检查数据集..."):
                st.session_state.dataset_report = inspect_dataset(draft["data_root"], labeled_percent=draft.get("labeled_percent", 10))
        report = st.session_state.dataset_report
        if report:
            st.caption(report.get("format", "UA-MT LA"))
            st.metric("Train cases", report["train_cases"])
            st.metric("Test cases", report["test_cases"])
            if report.get("labeled_cases") is not None:
                st.metric("Labeled / Unlabeled", f"{report['labeled_cases']} / {report['unlabeled_cases']}")
            if report["issues"]:
                for issue in report["issues"]:
                    st.warning(issue)
            else:
                st.success("数据目录结构检查通过。")
            if report["samples"]:
                st.json(report["samples"][0])
    with c2:
        cases = list_cases(draft["data_root"], "test")
        if not cases:
            st.info("请选择 UA-MT 的 data 目录；软件会读取 train.list、test.list 和病例 mri_norm2.h5。")
            return
        case = st.selectbox("病例", cases, key="preview_case")
        image, label = cached_load_case(draft["data_root"], case)
        axis_name = st.radio("切面", ["Axial", "Coronal", "Sagittal"], horizontal=True)
        axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[axis_name]
        max_slice = image.shape[axis] - 1
        index = st.slider("Slice", 0, max_slice, max_slice // 2)
        img_slice = np.take(image, index, axis=axis)
        lab_slice = np.take(label, index, axis=axis)
        cols = st.columns(3)
        cols[0].image(normalize_slice(img_slice), caption="MRI", clamp=True, use_container_width=True)
        cols[1].image((lab_slice > 0).astype(np.uint8) * 255, caption="Ground Truth", clamp=True, use_container_width=True)
        cols[2].image(overlay_slice(img_slice, lab_slice), caption="Overlay", clamp=True, use_container_width=True)

        st.markdown("#### 3D MRI / Ground Truth")
        st.caption("可旋转、缩放和平移。MRI 使用降采样体渲染以保证交互流畅；不会修改原始数据或训练输入。")
        preview_3d = mri_gt_3d_figure(image, label)
        st.plotly_chart(
            preview_3d,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )


def load_history(output):
    path = Path(output) / "history.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


@st.fragment(run_every="2s")
def render_training(draft):
    st.subheader("训练")
    st.caption("训练状态、Loss 曲线和日志每 2 秒自动更新。")
    status = JOB_MANAGER.get_status()
    top = st.columns(2)
    spec = get_model_spec(draft.get("model", "vnet"))
    start_disabled = status.get("state") in {"starting", "running", "finalizing"} or bool(status.get("process_alive"))
    if top[0].button("开始训练", type="primary", disabled=start_disabled, use_container_width=True):
        try:
            meta = JOB_MANAGER.start_training(draft)
            st.success(f"训练已启动：{meta['job_id']}")
            st.rerun()
        except Exception as exc:
            st.error(f"启动失败：{exc}")
    if top[1].button(
        "停止训练",
        disabled=status.get("state") not in {"starting", "running", "finalizing"},
        use_container_width=True,
    ):
        st.write(JOB_MANAGER.stop_training())
        st.rerun()

    status = JOB_MANAGER.get_status()
    state = status.get("state", "idle")
    st.write(f"**状态：** {state}")
    if state == "finalizing":
        st.info("训练已结束，正在释放 DataLoader / CUDA 资源；资源释放完成后会自动变为 completed，再次训练按钮才会启用。")
    if state != "idle":
        progress = float(status.get("progress", 0.0))
        st.progress(min(max(progress, 0.0), 1.0))
        cols = st.columns(5)
        cols[0].metric("Iteration", f"{status.get('iteration', 0)}/{status.get('max_iterations', '?')}")
        cols[1].metric("Best Dice", f"{status.get('best_dice', 0.0):.4f}")
        cols[2].metric("GPU workers", status.get("world_size", 1))
        cols[3].metric("Global batch", status.get("global_batch_size", "-"))
        cols[4].metric("samples/s", f"{status.get('samples_per_second', 0.0):.2f}")
        losses = status.get("losses", {})
        if losses:
            st.json(losses)
        validation = status.get("validation")
        if validation:
            vcols = st.columns(4)
            vcols[0].metric("Dice", f"{validation.get('dice', 0):.4f}")
            vcols[1].metric("Jaccard", f"{validation.get('jaccard', 0):.4f}")
            vcols[2].metric("ASD", f"{validation.get('asd', 0):.3f}")
            vcols[3].metric("HD95", f"{validation.get('hd95', 0):.3f}")
        if status.get("output"):
            history = load_history(status["output"])
            train_rows = [r for r in history if r.get("type") == "train" and "total" in r.get("losses", {})]
            if train_rows:
                frame = pd.DataFrame({
                    "iteration": [r["iteration"] for r in train_rows],
                    "total_loss": [r["losses"]["total"] for r in train_rows],
                })
                st.line_chart(frame, x="iteration", y="total_loss")
        if state == "failed":
            st.error(status.get("error", "训练失败"))
        with st.expander("训练日志"):
            st.code(JOB_MANAGER.tail_log(lines=120) or "暂无日志", language="text")


def render_inference(draft):
    st.subheader("3D 分割推理")
    spec = get_model_spec(draft.get("model", "vnet"))
    latest = JOB_MANAGER.latest_job()
    default_checkpoint = str(Path(latest["output"]) / "best.pt") if latest else ""
    checkpoint = path_input(
        "Checkpoint",
        key="infer_checkpoint",
        value=default_checkpoint,
        mode="file",
        file_filter="Checkpoint (*.pt *.pth *.ckpt *.tar);;All files (*)",
        help_text="点击 Open Files 从服务器选择用于推理的 checkpoint。",
    )
    cases = list_cases(draft["data_root"], "test")
    if not cases:
        st.info("test.list 中没有可用病例。")
        return
    case = st.selectbox("测试病例", cases, key="infer_case")
    if st.button("运行 3D 分割", type="primary"):
        try:
            with st.spinner("正在执行滑窗 3D 推理..."):
                st.session_state.inference_result = run_case_inference(
                    draft, checkpoint, case, WORKSPACE / "inference"
                )
        except Exception as exc:
            st.error(f"推理失败：{exc}")
    result = st.session_state.inference_result
    if result and result.get("case") == case:
        metrics = result["metrics"]
        ckpt = result.get("checkpoint", {})
        if ckpt:
            st.caption(
                f"实际加载 checkpoint: {ckpt.get('path', '')} | "
                f"iteration={ckpt.get('iteration', 0)} | "
                f"best_dice={ckpt.get('best_score', 0.0):.4f}"
            )
            if ckpt.get("missing_keys") or ckpt.get("unexpected_keys"):
                st.warning(
                    f"Checkpoint 参数并非完全匹配：missing={ckpt.get('missing_keys', [])}, "
                    f"unexpected={ckpt.get('unexpected_keys', [])}"
                )
        cols = st.columns(4)
        for col, key in zip(cols, ["dice", "jaccard", "asd", "hd95"]):
            col.metric(key.upper(), f"{metrics[key]:.4f}")
        raw_metrics = result.get("raw_metrics")
        if raw_metrics:
            st.caption(
                "LCC 前(raw): "
                f"Dice={raw_metrics['dice']:.4f}, "
                f"Jaccard={raw_metrics['jaccard']:.4f}, "
                f"ASD={raw_metrics['asd']:.4f}, "
                f"HD95={raw_metrics['hd95']:.4f}"
            )
        image = result["image"]
        label = result["label"]
        pred = result["prediction"]
        index = st.slider("Inference slice", 0, image.shape[2] - 1, image.shape[2] // 2)
        cols = st.columns(4)
        cols[0].image(normalize_slice(image[:, :, index]), caption="MRI", use_container_width=True)
        cols[1].image((label[:, :, index] > 0).astype(np.uint8) * 255, caption="GT", use_container_width=True)
        cols[2].image((pred[:, :, index] > 0).astype(np.uint8) * 255, caption="Prediction", use_container_width=True)
        cols[3].image(overlay_slice(image[:, :, index], pred[:, :, index]), caption="Overlay", use_container_width=True)

        st.markdown("#### 3D GT / Prediction")
        st.caption("GT 与 Prediction 位于两个独立 3D 坐标系；旋转、缩放或平移任意一侧，另一侧会同步保持完全相同的观察方向。")
        render_linked_segmentation_3d(label, pred)

        st.caption(f"输出目录：{result['output_dir']}")


def main():
    init_state()
    st.sidebar.markdown("## Medical AI Studio")
    workspace_mode = st.sidebar.radio(
        "工作区",
        ["3D 医学图像分割", "医疗多模态大模型"],
        horizontal=False,
        key="workspace_mode",
    )
    st.sidebar.divider()

    if workspace_mode == "3D 医学图像分割":
        st.title("Medical Image Segmentation Studio")
        st.caption(
            "Engineering-oriented 3D medical image segmentation · "
            "V-Net / CAT-Net · FSL / DNP · reproducible training/inference jobs"
        )
        draft, _ = render_sidebar(CONFIG_STORE, JOB_MANAGER, st.session_state.draft)
        tabs = st.tabs(["数据与预览", "训练", "推理"])
        with tabs[0]:
            render_workspace(draft)
        with tabs[1]:
            render_training(draft)
        with tabs[2]:
            render_inference(draft)
    else:
        st.title("医疗多模态大模型")
        if "foundation_draft" not in st.session_state:
            st.session_state.foundation_draft = FOUNDATION_CONFIG_STORE.read()
        foundation_draft, _ = render_foundation_sidebar(FOUNDATION_CONFIG_STORE, st.session_state.foundation_draft, PROJECT_ROOT)
        st.session_state.foundation_draft = foundation_draft
        render_foundation_workspace(foundation_draft, PROJECT_ROOT)


if __name__ == "__main__":
    main()
