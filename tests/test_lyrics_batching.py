import torch

from merge_emotion.data.lyrics import ensure_batch_encoding


def test_ensure_batch_encoding_adds_missing_batch_dimension():
    encoded = {
        "input_ids": torch.tensor([0, 10, 20, 2]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
    }
    batched = ensure_batch_encoding(encoded)
    assert batched["input_ids"].shape == (1, 4)
    assert batched["attention_mask"].shape == (1, 4)


def test_ensure_batch_encoding_preserves_existing_batch_dimension():
    encoded = {
        "input_ids": torch.tensor([[0, 10, 20, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    batched = ensure_batch_encoding(encoded)
    assert batched["input_ids"].shape == (1, 4)
    assert batched["attention_mask"].shape == (1, 4)
