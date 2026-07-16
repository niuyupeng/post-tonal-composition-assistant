"""Lightweight Transformer language model for score event tokens."""

from __future__ import annotations

from collections.abc import Callable

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
            raise ValueError(
                f"Input length {seq_len} exceeds model context {self.max_seq_len}; "
                "construct an explicit condition-preserving window before calling the model."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(self.norm(encoded))

    def _sampling_window(self, ids: list[int], prefix_length: int) -> list[int]:
        if len(ids) <= self.max_seq_len:
            return ids
        prefix_length = min(max(0, int(prefix_length)), self.max_seq_len - 1)
        tail_length = self.max_seq_len - prefix_length
        return ids[:prefix_length] + ids[-tail_length:]

    @staticmethod
    def _mask_logits(
        logits: torch.Tensor,
        allowed_ids: list[int] | None,
        eos_id: int,
        allow_eos: bool,
    ) -> torch.Tensor:
        masked = logits
        if allowed_ids is not None:
            valid = sorted({int(token_id) for token_id in allowed_ids if 0 <= int(token_id) < logits.shape[-1]})
            if not valid:
                raise ValueError("Grammar returned no valid next-token ids.")
            grammar_mask = torch.full_like(logits, float("-inf"))
            grammar_mask[..., valid] = logits[..., valid]
            masked = grammar_mask
        if not allow_eos:
            masked = masked.clone()
            masked[..., eos_id] = float("-inf")
        return masked

    @torch.no_grad()
    def sample(
        self,
        prefix_ids: list[int],
        eos_id: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 20,
        min_new_tokens: int = 0,
        allowed_token_ids_fn: Callable[[list[int]], list[int]] | None = None,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = list(prefix_ids)
        prefix_length = len(prefix_ids)
        for step in range(max(0, int(max_new_tokens))):
            window = self._sampling_window(ids, prefix_length)
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids, dtype=torch.bool)
            logits = self(input_ids, attention)[:, -1, :] / max(temperature, 1e-6)
            allowed = allowed_token_ids_fn(ids) if allowed_token_ids_fn is not None else None
            logits = self._mask_logits(logits, allowed, eos_id, step >= int(min_new_tokens))
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
        max_new_tokens: int | list[int] = 128,
        temperature: float = 1.0,
        top_k: int | None = 20,
        generators: list[torch.Generator | None] | None = None,
        use_amp: bool = False,
        min_new_tokens: int | list[int] = 0,
        allowed_token_ids_fns: list[Callable[[list[int]], list[int]] | None] | None = None,
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
        if isinstance(max_new_tokens, int):
            max_token_counts = [max(0, int(max_new_tokens))] * len(prefix_ids)
        else:
            max_token_counts = [max(0, int(value)) for value in max_new_tokens]
        if len(max_token_counts) != len(prefix_ids):
            raise ValueError("Expected one max_new_tokens value per sampling prefix.")
        if isinstance(min_new_tokens, int):
            min_token_counts = [max(0, int(min_new_tokens))] * len(prefix_ids)
        else:
            min_token_counts = [max(0, int(value)) for value in min_new_tokens]
        if len(min_token_counts) != len(prefix_ids):
            raise ValueError("Expected one min_new_tokens value per sampling prefix.")
        if allowed_token_ids_fns is None:
            allowed_token_ids_fns = [None] * len(prefix_ids)
        if len(allowed_token_ids_fns) != len(prefix_ids):
            raise ValueError("Expected one grammar callback per sampling prefix.")

        self.eval()
        device = next(self.parameters()).device
        sequences = [list(prefix) for prefix in prefix_ids]
        prefix_lengths = [len(prefix) for prefix in prefix_ids]
        generated_counts = [0] * len(sequences)
        active = [index for index, budget in enumerate(max_token_counts) if budget > 0]
        amp_enabled = bool(use_amp and device.type == "cuda")

        for _ in range(max(max_token_counts, default=0)):
            if not active:
                break
            windows = [
                self._sampling_window(sequences[index], prefix_lengths[index])
                for index in active
            ]
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
            for row, sequence_index in enumerate(active):
                callback = allowed_token_ids_fns[sequence_index]
                allowed = callback(sequences[sequence_index]) if callback is not None else None
                logits[row] = self._mask_logits(
                    logits[row],
                    allowed,
                    eos_id,
                    generated_counts[sequence_index] >= min_token_counts[sequence_index],
                )
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
                generated_counts[sequence_index] += 1
                if next_id != eos_id and generated_counts[sequence_index] < max_token_counts[sequence_index]:
                    remaining.append(sequence_index)
            active = remaining
        return sequences
