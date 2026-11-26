from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_server_picker_is_streamlit_native_and_has_no_qt_x11_dependency():
    source = Path("studio/ui/path_widgets.py").read_text(encoding="utf-8")
    assert '@st.dialog("选择服务器目录"' in source
    assert '@st.dialog("选择服务器文件"' in source
    assert '选择当前目录' in source
    assert '↑ 上一级' in source
    assert ':hover::after' in source
    assert 'PySide6' not in source
    assert 'QFileDialog' not in source
    assert 'DISPLAY' not in source
    assert not Path("studio/native_dialog.py").exists()


def test_base_dependencies_no_longer_require_qt():
    with Path("pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    assert cfg["project"]["version"] == "1.2.4"
    deps = cfg["project"]["dependencies"]
    assert all("PySide" not in item and "PyQt" not in item for item in deps)
    req = Path("requirements.txt").read_text(encoding="utf-8")
    env = Path("environment.yml").read_text(encoding="utf-8")
    assert "PySide" not in req and "PyQt" not in req
    assert "PySide" not in env and "PyQt" not in env


def test_picker_does_not_mutate_instantiated_jump_widget_state():
    source = Path("studio/ui/path_widgets.py").read_text(encoding="utf-8")
    # Regression for StreamlitWidgetAlreadyInstantiatedError: the old picker
    # instantiated a text_input with this key and then reassigned it while
    # handling a folder button click in the same run.
    assert "picker_jump" not in source
    assert "跳转到" not in source
