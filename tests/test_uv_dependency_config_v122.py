from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_uv_linux_only_and_serving_extra_is_resolvable_shape():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)

    assert cfg["project"]["version"] == "1.2.4"
    assert cfg["tool"]["uv"]["environments"] == ["sys_platform == 'linux'"]
    serving = cfg["project"]["optional-dependencies"]["serving"]
    assert serving == ["vllm>=0.22; sys_platform == 'linux'"]
    assert all("vllm-bnb-plugin" not in item for item in serving)


def test_requirements_serving_has_no_removed_plugin():
    root = Path(__file__).resolve().parents[1]
    text = (root / "requirements-serving.txt").read_text(encoding="utf-8")
    assert "vllm-bnb-plugin" not in text
    assert "vllm>=0.22" in text
