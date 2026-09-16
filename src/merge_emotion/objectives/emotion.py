from __future__ import annotations
import torch.nn.functional as F


def emotion_loss(outputs, labels, modality, lambda_uni=0.5, label_smoothing=0.0):
    if modality == "audio":
        return F.cross_entropy(outputs["audio_logits"], labels, label_smoothing=label_smoothing)
    if modality == "lyrics":
        return F.cross_entropy(outputs["lyrics_logits"], labels, label_smoothing=label_smoothing)
    main = F.cross_entropy(outputs["multimodal_logits"], labels, label_smoothing=label_smoothing)
    aux = 0.5 * (F.cross_entropy(outputs["audio_logits"], labels, label_smoothing=label_smoothing) + F.cross_entropy(outputs["lyrics_logits"], labels, label_smoothing=label_smoothing))
    return main + lambda_uni * aux
