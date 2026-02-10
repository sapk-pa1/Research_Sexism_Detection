"""Backward-compatible exports for the adaptive student network."""

from sexism_distillation.losses.distillation import DistillationLoss, DistillationLossConfig
from sexism_distillation.models.student import AdaptiveStudentConfig, AdaptiveStudentNetwork

__all__ = [
    "AdaptiveStudentConfig",
    "AdaptiveStudentNetwork",
    "DistillationLoss",
    "DistillationLossConfig",
]
