from __future__ import annotations
from torch import nn
from .audio_encoder import AudioEncoder
from .lyrics_encoder import LyricsEncoder
from .fusion import GatedCrossModalFusion


class EmotionAlignmentModel(nn.Module):
    def __init__(self, audio_cfg, lyrics_cfg, multimodal_cfg):
        super().__init__(); h=multimodal_cfg["hidden_dim"]
        self.audio_encoder=AudioEncoder(audio_cfg); self.lyrics_encoder=LyricsEncoder(lyrics_cfg)
        self.audio_head=nn.Linear(h,multimodal_cfg["num_classes"]); self.lyrics_head=nn.Linear(h,multimodal_cfg["num_classes"])
        self.fusion=GatedCrossModalFusion(h,multimodal_cfg["attention_heads"],multimodal_cfg["cross_attention_dropout"],multimodal_cfg["fusion_layers"],multimodal_cfg["ffn_dim"]); self.multimodal_head=nn.Linear(h,multimodal_cfg["num_classes"])

    def representation_parameters(self):
        for module in [self.audio_encoder,self.lyrics_encoder,self.fusion]:
            yield from module.parameters()

    def forward(self,batch,modality="multimodal"):
        out={}
        if modality in {"audio","multimodal"}:
            at, ar=self.audio_encoder(batch["audio_features"],batch["audio_mask"]); out.update(audio_tokens=at,audio_repr=ar,audio_logits=self.audio_head(ar))
        if modality in {"lyrics","multimodal"}:
            lt, lr=self.lyrics_encoder(batch["lyrics_features"],batch["lyrics_mask"]); out.update(lyrics_tokens=lt,lyrics_repr=lr,lyrics_logits=self.lyrics_head(lr))
        if modality=="multimodal":
            mr=self.fusion(out["audio_tokens"],out["lyrics_tokens"],batch["audio_mask"],batch["lyrics_mask"]); out.update(multimodal_repr=mr,multimodal_logits=self.multimodal_head(mr))
        return out
