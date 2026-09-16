from merge_emotion.config import deep_merge


def test_deep_merge_preserves_nested_values():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


from pathlib import Path
from merge_emotion.config import compose_config


def test_pipeline_config_is_self_contained():
    root = Path(__file__).resolve().parents[1]
    config = compose_config(root / "configs/pipeline_experiments/cspa_affect.yaml", root)
    assert config["dataset"]["manifest"] == "data/processed/pairs_70-15-15.csv"
    assert config["audio_model"]["input_dim"] == 768
    assert config["lyrics_model"]["max_tokens"] == 8
    assert config["multimodal"]["fusion_layers"] == 2
    assert config["cspa"]["enabled"] is True
    assert config["affect_aware"]["enabled"] is True
