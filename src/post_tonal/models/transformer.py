"""Lightweight Transformer language model for score event tokens."""

from __future__ import annotations

import torch
from torch import nn


class PostTonalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        layers: int = 6,
        heads: int = 6,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_seq_len]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(self.norm(encoded))

    @torch.no_grad()
    def sample(
        self,
        prefix_ids: list[int],
        eos_id: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 20,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = list(prefix_ids)
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids[-self.max_seq_len :]], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids, dtype=torch.bool)
            logits = self(input_ids, attention)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if next_id == eos_id:
                break
        return ids

    @torch.no_grad()
    def sample_batch(
        self,
        prefix_ids: list[list[int]],
        eos_id: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 20,
        generators: list[torch.Generator | None] | None = None,
        use_amp: bool = False,
    ) -> list[list[int]]:
        """Sample variable-length continuations with one RNG stream per sequence."""
        if not prefix_ids:
            return []
        if any(not prefix for prefix in prefix_ids):
            raise ValueError("Every sampling prefix must contain at least one token.")
        if generators is None:
            generators = [None] * len(prefix_ids)
        if len(generators) != len(prefix_ids):
            raise ValueError("Expected one generator per sampling prefix.")

        self.eval()
        device = next(self.parameters()).device
        sequences = [list(prefix) for prefix in prefix_ids]
        active = list(range(len(sequences)))
        amp_enabled = bool(use_amp and device.type == "cuda")

        for _ in range(max_new_tokens):
            if not active:
                break
            windows = [sequences[index][-self.max_seq_len :] for index in active]
            lengths = torch.tensor([len(window) for window in windows], dtype=torch.long, device=device)
            width = int(lengths.max().item())
            input_ids = torch.zeros((len(active), width), dtype=torch.long, device=device)
            attention = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, window in enumerate(windows):
                length = len(window)
                input_ids[row, :length] = torch.tensor(window, dtype=torch.long, device=device)
                attention[row, :length] = True

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                all_logits = self(input_ids, attention)
                logits = all_logits[
                    torch.arange(len(active), device=device),
                    lengths - 1,
                ] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probabilities = torch.softmax(logits, dim=-1)

            remaining: list[int] = []
            for row, sequence_index in enumerate(active):
                next_id = int(
                    torch.multinomial(
                        probabilities[row],
                        num_samples=1,
                        generator=generators[sequence_index],
                    ).item()
                )
                sequences[sequence_index].append(next_id)
                if next_id != eos_id:
                    remaining.append(sequence_index)
            active = remaining
        return sequences
