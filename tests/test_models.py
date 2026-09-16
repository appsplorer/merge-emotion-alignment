import torch
from merge_emotion.models.multimodal import EmotionAlignmentModel


def make_model():
    audio = {"input_dim": 8, "hidden_dim": 16, "transformer_layers": 1, "attention_heads": 4, "ffn_dim": 32, "dropout": 0.0, "max_tokens": 6}
    lyrics = {"input_dim": 8, "hidden_dim": 16, "transformer_layers": 1, "attention_heads": 4, "ffn_dim": 32, "dropout": 0.0, "max_tokens": 8}
    mm = {"hidden_dim": 16, "attention_heads": 4, "cross_attention_dropout": 0.0, "fusion_layers": 1, "ffn_dim": 32, "num_classes": 4}
    return EmotionAlignmentModel(audio, lyrics, mm)


def test_multimodal_forward_shapes():
    model = make_model()
    batch = {"audio_features": torch.randn(3, 6, 8), "audio_mask": torch.ones(3, 6, dtype=torch.bool), "lyrics_features": torch.randn(3, 8, 8), "lyrics_mask": torch.ones(3, 8, dtype=torch.bool)}
    out = model(batch, "multimodal")
    assert out["audio_repr"].shape == (3, 16)
    assert out["lyrics_repr"].shape == (3, 16)
    assert out["multimodal_logits"].shape == (3, 4)
