from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCNNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class LightweightAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        return self.norm(x + self.dropout(out)), attn_weights


class KeyAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Parameter(torch.randn(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = torch.tanh(self.proj(x))
        scores = torch.einsum("bsh,h->bs", keys, self.query)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bs,bsh->bh", weights, x)
        return pooled, weights


@dataclass
class AdaptiveStudentConfig:
    vocab_size: int
    num_classes: int
    hidden_size: int = 256
    max_position_embeddings: int = 512
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    cnn_kernel_1: int = 3
    cnn_kernel_2: int = 5
    use_skip_connection: bool = True


class AdaptiveStudentNetwork(nn.Module):
    """Embedding -> CNN -> CNN -> Lightweight attention -> FFN -> Key attention -> Output."""

    def __init__(self, config: AdaptiveStudentConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.cnn1 = LocalCNNBlock(config.hidden_size, kernel_size=config.cnn_kernel_1, dropout=config.dropout)
        self.cnn2 = LocalCNNBlock(config.hidden_size, kernel_size=config.cnn_kernel_2, dropout=config.dropout)
        self.light_attention = LightweightAttention(config.hidden_size, num_heads=config.num_heads, dropout=config.dropout)

        ffn_hidden = config.hidden_size * config.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_hidden, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)

        self.key_attention = KeyAttentionLayer(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        embedding_out = x

        x = self.cnn1(x)
        x = self.cnn2(x)

        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x, attn_weights = self.light_attention(x, key_padding_mask=key_padding_mask)

        if self.config.use_skip_connection:
            x = self.ffn_norm(x + self.ffn(x))
        else:
            x = self.ffn_norm(self.ffn(x))

        pooled, key_weights = self.key_attention(x, attention_mask=attention_mask)
        logits = self.classifier(pooled)

        if return_aux:
            return {
                "logits": logits,
                "hidden_states": [embedding_out, x],
                "attention_weights": attn_weights,
                "key_attention": key_weights,
            }
        return logits
