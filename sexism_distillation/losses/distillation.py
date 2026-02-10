from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationLossConfig:
    alpha_logits: float = 0.5
    alpha_hidden: float = 0.3
    alpha_attention: float = 0.2
    temperature: float = 2.0


class DistillationLoss(nn.Module):
    """KD loss = logits KL + hidden-state alignment + selective attention loss + CE."""

    def __init__(self, config: DistillationLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_attention: Optional[torch.Tensor] = None,
        teacher_attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(student_logits, labels)

        t = self.config.temperature
        kd_targets = F.softmax(teacher_logits / t, dim=-1)
        kd_preds = F.log_softmax(student_logits / t, dim=-1)
        logits_kd = F.kl_div(kd_preds, kd_targets, reduction="batchmean") * (t**2)

        hidden_loss = student_logits.new_tensor(0.0)
        if student_hidden is not None and teacher_hidden is not None:
            if teacher_hidden.size(-1) != student_hidden.size(-1):
                teacher_hidden = teacher_hidden[..., : student_hidden.size(-1)]
            hidden_loss = F.mse_loss(student_hidden, teacher_hidden)

        attention_loss = student_logits.new_tensor(0.0)
        if student_attention is not None and teacher_attention is not None:
            if teacher_attention.dim() > student_attention.dim():
                teacher_attention = teacher_attention.mean(dim=1)
            attention_loss = F.mse_loss(student_attention, teacher_attention)

        return (
            (1.0 - self.config.alpha_logits) * ce_loss
            + self.config.alpha_logits * logits_kd
            + self.config.alpha_hidden * hidden_loss
            + self.config.alpha_attention * attention_loss
        )
