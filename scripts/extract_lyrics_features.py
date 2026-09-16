#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from merge_emotion.data.lyrics import chunk_token_ids, ensure_batch_encoding

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/pairs_70-15-15.csv")
    p.add_argument("--model", default="/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache/FacebookAI__roberta-base")
    p.add_argument("--output", default="cache/lyrics/roberta_base.pt")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(PROJECT_ROOT / args.manifest).drop_duplicates("lyric_song")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        add_pooling_layer=False,
    ).to(device).eval()
    ids, features, masks = [], [], []
    hidden_dim = int(model.config.hidden_size)
    empty_lyrics = 0
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="lyrics"):
        text = Path(row["lyric_path"]).read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            empty_lyrics += 1
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = chunk_token_ids(token_ids, args.chunk_tokens, args.max_chunks)
        song_features = torch.zeros(args.max_chunks, hidden_dim)
        song_mask = torch.zeros(args.max_chunks, dtype=torch.bool)
        for i, chunk in enumerate(chunks):
            encoded = tokenizer.prepare_for_model(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=args.chunk_tokens + tokenizer.num_special_tokens_to_add(),
                return_tensors="pt",
            )
            encoded = ensure_batch_encoding(encoded)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(**encoded, return_dict=True)
                    mask = encoded.get(
                        "attention_mask",
                        torch.ones(out.last_hidden_state.shape[:2], device=device),
                    ).unsqueeze(-1)
                    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            song_features[i] = pooled.squeeze(0).float().cpu()
            song_mask[i] = True
        ids.append(str(row["lyric_song"]))
        features.append(song_features)
        masks.append(song_mask)
    stacked = torch.stack(features)
    if not torch.isfinite(stacked).all():
        raise RuntimeError("Non-finite RoBERTa features detected")
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ids": ids, "features": stacked, "mask": torch.stack(masks), "model": args.model},
        output,
    )
    print("Saved", output, stacked.shape, "empty_lyrics=", empty_lyrics)


if __name__ == "__main__":
    main()
