from __future__ import annotations
import math
import torch


def segment_waveform(waveform: torch.Tensor, sample_rate: int, segment_seconds: float, max_segments: int):
    segment_samples = int(round(sample_rate * segment_seconds))
    needed = segment_samples * max_segments
    waveform = waveform[:needed]
    segments = []
    mask = []
    for i in range(max_segments):
        start = i * segment_samples
        chunk = waveform[start:start + segment_samples]
        if chunk.numel() == 0:
            chunk = torch.zeros(segment_samples, dtype=waveform.dtype)
            valid = False
        else:
            valid = True
            if chunk.numel() < segment_samples:
                chunk = torch.nn.functional.pad(chunk, (0, segment_samples - chunk.numel()))
        segments.append(chunk)
        mask.append(valid)
    return torch.stack(segments), torch.tensor(mask, dtype=torch.bool)
