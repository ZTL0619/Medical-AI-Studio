# Changelog

## GitHub source distribution

- Removed obsolete resident Transformers and native Qt dialog implementations.
- Removed machine-specific model paths and excluded private runtime artifacts.
- Preserved segmentation, training and retrieval algorithms; updated regression tests and documentation.

## Medical VLM update patch

- Transformers 兼容推理采用按请求隔离子进程；启用 GraphRAG 时单次请求复用同一模型，退出后释放 GPU。
- 统一医疗多模态模型 Registry：补回 `MedGemma-4B`，保留 `Lingshu-7B` 与自定义本地 VLM。
- SFT 改为 assistant-only loss：system / image / user prompt / padding 全部 mask，只监督 assistant answer。
- 增加验证集：可单独指定验证 JSONL；留空时按固定 seed 自动划分。
- 训练启用 validation loss，并按最低 `eval_loss` 选择最佳 checkpoint，导出 `adapter_best`。
- UI 中显示 train loss、validation loss、best validation loss 和 best adapter 路径。
- 保留 v1.2.4 的服务器路径选择器与原有 3-D 分割功能。

## v1.2.4

- 修复服务器路径选择器的 `StreamlitWidgetAlreadyInstantiatedError`。
- 删除弹窗内部“跳转到路径”输入框及其 `picker_jump` session-state，同一轮运行不再修改已实例化 widget 的 key。
- 路径弹窗收敛为：当前路径、上一级、Home、目录列表、选择当前目录、取消。
- 主路径输入框仍可直接手工粘贴任意服务器路径，并保留完整路径悬停提示。
- 不改动 V-Net/CAT-Net/DNP/FSL、大模型、多 GPU、QLoRA、vLLM、GraphRAG 接口。

## v1.2.3

- Removed the Qt/PySide6 desktop chooser and all X11/DISPLAY requirements.
- Added a pure Streamlit modal server filesystem browser for directory and file selection.
- Added parent-directory, Home, direct-path navigation, directory selection, and file-extension filtering.
- Kept manual path entry and full-path hover tooltips.
- Reused the same picker for segmentation data, model paths, adapters, fine-tuning datasets, and resume checkpoints.
- Preserved all v1.2.2 segmentation, Medical VLM, multi-GPU, PEFT, vLLM, and GraphRAG code.
- Added regression tests preventing Qt/X11 dependencies from returning.

## v1.2.2

- Fixed `uv` dependency resolution failure caused by the invalid `vllm-bnb-plugin>=0.1` requirement.
- Removed `vllm-bnb-plugin`; the current serving code uses vLLM's native bitsandbytes quantization path and does not import that plugin.
- Marked the optional vLLM serving dependency as Linux-only.
- Added `tool.uv.environments = ["sys_platform == 'linux'"]` so `uv` no longer tries to solve the CUDA/vLLM stack for macOS.
- Preserved all v1.2.1 UI, segmentation, multi-GPU, PEFT, vLLM, and GraphRAG code unchanged.

## v1.2.1

- Rebuilt the Medical VLM UI as an end-user workspace: sidebar configuration + one multimodal chat panel.
- Removed developer-facing Overview / PEFT / vLLM / Benchmark / GraphRAG tabs from the main page.
- Kept benchmark/profiling and GraphRAG contracts as backend engineering infrastructure.
- Fixed the Streamlit `disabled=None` crash by enforcing strict boolean training state.
- Replaced the web directory browser with a Qt native file/directory chooser suitable for SSH X11/XLaunch.
- Kept full-path hover tooltip for path fields.
- Replaced fixed “dual GPU” assumptions in UI with dynamic multi-GPU selection and dynamic `world_size`.
- Multi-GPU DDP/NCCL and vLLM DP/TP command construction now explicitly tested with more than two GPUs.
- Added model/data/adapter native path selection where applicable.
- Added a chat-style Image + Text → Text UI with conversation history.
- Preserved the v1.1.1 segmentation algorithms, inference, 3D visualization and GPU cleanup lifecycle.

## v1.2.0

- Added Medical VLM engineering branch with LoRA/QLoRA, Accelerate DDP, optional ZeRO-2, vLLM, evaluation/profiling and Hybrid GraphRAG contracts.
- Removed the SAM placeholder branch.

## Medical knowledge builder UI
- Added build/import-first workflow for ④ 医学知识增强.
- Added conservative PDF cleaning, scanned-PDF detection/skipping, mixed-PDF page filtering, live build progress and detailed reports.
- Added portable `snippets.jsonl` + `build_report.json` and knowledge-base validation before import.
