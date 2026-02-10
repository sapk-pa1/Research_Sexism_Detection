from __future__ import annotations

import random
from dataclasses import dataclass

from sexism_distillation.models.student import AdaptiveStudentConfig


@dataclass
class NASSearchSpace:
    hidden_sizes: tuple[int, ...] = (128, 192, 256)
    num_heads: tuple[int, ...] = (2, 4, 8)
    cnn_kernels: tuple[int, ...] = (3, 5, 7)
    use_skip_connection: tuple[bool, ...] = (True, False)


class NASController:
    """Simple random-search NAS controller for student architecture."""

    def __init__(self, search_space: NASSearchSpace) -> None:
        self.search_space = search_space

    def sample_config(self, vocab_size: int, num_classes: int) -> AdaptiveStudentConfig:
        hidden = random.choice(self.search_space.hidden_sizes)
        heads = [h for h in self.search_space.num_heads if hidden % h == 0]
        if not heads:
            heads = (1,)

        return AdaptiveStudentConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            hidden_size=hidden,
            num_heads=random.choice(tuple(heads)),
            cnn_kernel_1=random.choice(self.search_space.cnn_kernels),
            cnn_kernel_2=random.choice(self.search_space.cnn_kernels),
            use_skip_connection=random.choice(self.search_space.use_skip_connection),
        )
