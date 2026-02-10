from __future__ import annotations

import time


def accuracy_score(preds, labels) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / total


def model_size_mb(model) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024**2)


def measure_latency_ms(model, sample_batch, warmup: int = 3, iters: int = 20) -> float:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for latency measurement") from exc

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**sample_batch)

        start = time.perf_counter()
        for _ in range(iters):
            _ = model(**sample_batch)
        end = time.perf_counter()

    return ((end - start) / iters) * 1000.0
