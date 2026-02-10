from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class RuleBasedCollapsingConfig:
    importance_threshold: float = 0.15
    gradient_sensitivity_threshold: float = 0.10
    flops_threshold: float = 0.20


def compute_importance_scores(weight_norms: Dict[str, float]) -> Dict[str, float]:
    total = sum(weight_norms.values()) or 1.0
    return {k: v / total for k, v in weight_norms.items()}


def compute_gradient_sensitivity(grad_norms: Dict[str, float]) -> Dict[str, float]:
    max_norm = max(grad_norms.values()) if grad_norms else 1.0
    return {k: (v / max_norm if max_norm > 0 else 0.0) for k, v in grad_norms.items()}


def estimate_flops_ratio(base_flops: float, candidate_flops: float) -> float:
    if base_flops <= 0:
        return 0.0
    return 1.0 - (candidate_flops / base_flops)


def should_keep_block(
    block_name: str,
    importance_scores: Dict[str, float],
    gradient_sensitivity: Dict[str, float],
    flops_saving_ratio: float,
    config: RuleBasedCollapsingConfig,
) -> bool:
    importance = importance_scores.get(block_name, 1.0)
    sensitivity = gradient_sensitivity.get(block_name, 1.0)

    if flops_saving_ratio < config.flops_threshold:
        return True
    if importance < config.importance_threshold and sensitivity < config.gradient_sensitivity_threshold:
        return False
    return True


def filter_blocks(
    blocks: Iterable[str],
    importance_scores: Dict[str, float],
    gradient_sensitivity: Dict[str, float],
    flops_saving_ratio: float,
    config: RuleBasedCollapsingConfig,
) -> list[str]:
    return [
        b
        for b in blocks
        if should_keep_block(b, importance_scores, gradient_sensitivity, flops_saving_ratio, config)
    ]
