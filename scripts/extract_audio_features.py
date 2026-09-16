#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from merge_emotion.data.audio import segment_waveform

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_audio(path, target_sr):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/pairs_70-15-15.csv")
    p.add_argument("--model", default="/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/m-a-p__MERT-v1-95M")
    p.add_argument("--output", default="cache/audio/mert_v1_95m.pt")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--max-segments", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(PROJECT_ROOT / args.manifest).drop_duplicates("audio_song")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True).to(device).eval()
    ids, all_features, all_masks = [], [], []
    pending_waves, pending_meta = [], []

    def flush():
        if not pending_waves:
            return
        inputs = processor([x.numpy() for x in pending_waves], sampling_rate=args.sample_rate, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            out = model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
            hidden = out.last_hidden_state.mean(dim=1).cpu()
        by_song = {}
        for feat, (song_id, segment_index) in zip(hidden, pending_meta):
            by_song.setdefault(song_id, {})[segment_index] = feat
        for song_id in sorted(by_song, key=lambda x: next(i for i, m in enumerate(pending_meta) if m[0] == x)):
            segment_map = by_song[song_id]
            features = torch.stack([segment_map[i] for i in sorted(segment_map)])
            mask = torch.ones(features.size(0), dtype=torch.bool)
            all_features.append(features)
            all_masks.append(mask)
            ids.append(song_id)
        pending_waves.clear(); pending_meta.clear()

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="audio"):
        waveform = load_audio(row["audio_path"], args.sample_rate)
        segments, mask = segment_waveform(waveform, args.sample_rate, args.segment_seconds, args.max_segments)
        valid_segments = [seg for seg, valid in zip(segments, mask.tolist()) if valid]
        for index, seg in enumerate(valid_segments):
            pending_waves.append(seg)
            pending_meta.append((str(row["audio_song"]), index))
        if len(pending_waves) >= args.batch_size * args.max_segments:
            flush()
    flush()
    max_len = args.max_segments
    dim = all_features[0].size(-1)
    features = torch.zeros(len(ids), max_len, dim)
    masks = torch.zeros(len(ids), max_len, dtype=torch.bool)
    for i, (feat, mask) in enumerate(zip(all_features, all_masks)):
        n = min(max_len, feat.size(0))
        features[i, :n] = feat[:n]
        masks[i, :n] = mask[:n]
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "features": features, "mask": masks, "model": args.model, "sample_rate": args.sample_rate}, output)
    print("Saved", output, features.shape)


if __name__ == "__main__":
    main()
