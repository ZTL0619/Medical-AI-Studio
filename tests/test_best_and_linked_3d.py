from pathlib import Path


def test_runner_checks_best_on_every_validation():
    source = Path("ssnet_final/engine/runner.py").read_text(encoding="utf-8")
    assert 'best_updated = (not best_initialized) or (score > best_score)' in source
    assert 'save_checkpoint(\n                                best_path' in source
    assert 'best_initialized = True' in source
    assert 'Do not fabricate best.pt at training shutdown' in source


def test_inference_3d_uses_two_linked_scenes():
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'specs=[[{"type": "scene"}, {"type": "scene"}]]' in source
    assert 'scene2=scene_common' in source
    assert "gd.on('plotly_relayout'" in source
    assert "target = scene1Changed ? 'scene2' : 'scene'" in source
