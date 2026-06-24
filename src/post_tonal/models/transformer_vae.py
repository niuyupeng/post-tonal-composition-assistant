"""Optional Transformer-VAE baseline components.

The main experiments use `PostTonalTransformer`. This module keeps a compact
VAE variant available for ablations without requiring a separate training path.
"""

from __future__ import annotations

import torch
from torch import nn

from post_tonal.models.transformer import PostTonalTransformer


class TransformerVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        latent_size: int = 64,
        layers: int = 4,
        heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = PostTonalTransformer(vocab_size, hidden_size, layers, heads, max_seq_len, dropout)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.backbone(input_ids, attention_mask)
        pooled = self.backbone.token_embedding(input_ids).mean(dim=1)
        mu = self.mu(pooled)
        logvar = self.logvar(pooled)
        return logits, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
