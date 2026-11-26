from pathlib import Path


def test_training_marks_finalizing_before_completed_and_releases_cuda():
    source = Path("ssnet_final/engine/runner.py").read_text(encoding="utf-8")
    assert 'state="finalizing"' in source
    assert '_release_cuda_training_resources(context.device)' in source
    assert 'training_step = None' in source
    assert 'optimizer = None' in source
    assert 'eval_model = None' in source
    assert 'resources_released=True' in source
    assert source.index('state="finalizing"') < source.index('_release_cuda_training_resources(context.device)')
    assert source.index('_release_cuda_training_resources(context.device)') < source.index('resources_released=True')


def test_job_manager_blocks_overlap_and_ignores_zombies():
    source = Path("studio/job_manager.py").read_text(encoding="utf-8")
    assert 'proc.status() != psutil.STATUS_ZOMBIE' in source
    assert 'self._assert_previous_job_released()' in source
    assert 'status["state"] = "finalizing"' in source


def test_data_root_uses_server_path_widget_with_hover_tooltip():
    sidebar = Path("studio/ui/sidebar.py").read_text(encoding="utf-8")
    widget = Path("studio/ui/path_widgets.py").read_text(encoding="utf-8")
    assert 'path_input(' in sidebar
    assert '"数据目录"' in sidebar
    assert ':hover::after' in widget
    assert '@st.dialog("选择服务器目录"' in widget
    assert 'QFileDialog' not in widget
