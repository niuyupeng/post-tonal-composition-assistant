"""Twelve-tone row and serial-transformation utilities."""

from __future__ import annotations

import random
import re
from typing import Iterable


ROW_FORM_RE = re.compile(r"^(RI|P|R|I)(\d{1,2})$")


def generate_twelve_tone_row(seed: int | None = None, rng: random.Random | None = None) -> list[int]:
    local_rng = rng if rng is not None else random.Random(seed)
    row = list(range(12))
    local_rng.shuffle(row)
    return row


def is_valid_row(row: Iterable[int]) -> bool:
    values = [int(pc) % 12 for pc in row]
    return len(values) == 12 and sorted(values) == list(range(12))


def _validate(row: Iterable[int]) -> list[int]:
    values = [int(pc) % 12 for pc in row]
    if not is_valid_row(values):
        raise ValueError("Expected a valid twelve-tone row containing each pitch class once.")
    return values


def transpose_row_to_start(row: Iterable[int], n: int) -> list[int]:
    values = _validate(row)
    shift = (int(n) - values[0]) % 12
    return [(pc + shift) % 12 for pc in values]


def P(row: Iterable[int], n: int = 0) -> list[int]:
    return transpose_row_to_start(row, n)


def R(row: Iterable[int], n: int = 0) -> list[int]:
    return list(reversed(P(row, n)))


def I(row: Iterable[int], n: int = 0) -> list[int]:
    values = _validate(row)
    first = values[0]
    return [(int(n) - (pc - first)) % 12 for pc in values]


def RI(row: Iterable[int], n: int = 0) -> list[int]:
    return list(reversed(I(row, n)))


def parse_row_form(label: str) -> tuple[str, int]:
    match = ROW_FORM_RE.match(label.strip().upper())
    if not match:
        raise ValueError(f"Invalid row form label: {label!r}")
    form, transposition = match.group(1), int(match.group(2)) % 12
    return form, transposition


def row_form(row: Iterable[int], label: str) -> list[int]:
    form, transposition = parse_row_form(label)
    if form == "P":
        return P(row, transposition)
    if form == "R":
        return R(row, transposition)
    if form == "I":
        return I(row, transposition)
    if form == "RI":
        return RI(row, transposition)
    raise ValueError(f"Unsupported row form: {form}")


def row_order_accuracy(generated_pcs: Iterable[int], expected_row: Iterable[int]) -> float:
    """Position-wise accuracy against the expected row order.

    Extra generated pcs are ignored after the first aggregate. Missing notes count
    as incorrect positions.
    """

    expected = [int(pc) % 12 for pc in expected_row]
    if not expected:
        return 1.0
    generated = [int(pc) % 12 for pc in generated_pcs]
    correct = 0
    for idx, expected_pc in enumerate(expected):
        if idx < len(generated) and generated[idx] == expected_pc:
            correct += 1
    return correct / len(expected)


def cyclic_row_order_accuracy(generated_pcs: Iterable[int], expected_row: Iterable[int]) -> float:
    """Accuracy for longer passages that may cycle through the same row form."""

    expected = [int(pc) % 12 for pc in expected_row]
    generated = [int(pc) % 12 for pc in generated_pcs]
    if not expected:
        return 1.0
    if not generated:
        return 0.0
    correct = sum(1 for idx, pc in enumerate(generated) if pc == expected[idx % len(expected)])
    return correct / len(generated)


def serial_transformation_accuracy(
    generated_pcs: Iterable[int],
    source_row: Iterable[int],
    expected_label: str,
) -> float:
    """Exact requested-form accuracy over complete generated aggregates.

    This is deliberately stricter than cyclic row-order accuracy. Each complete
    non-overlapping 12-note block must equal the requested P/R/I/RI form.
    Partial terminal blocks do not count as completed transformations.
    """

    generated = [int(pc) % 12 for pc in generated_pcs]
    expected = row_form(source_row, expected_label)
    complete_blocks = len(generated) // 12
    if complete_blocks == 0:
        return 0.0
    correct = 0
    for block_index in range(complete_blocks):
        start = block_index * 12
        if generated[start : start + 12] == expected:
            correct += 1
    return correct / complete_blocks


def aggregate_completion_rate(generated_pcs: Iterable[int]) -> float:
    generated = {int(pc) % 12 for pc in generated_pcs}
    return len(generated) / 12.0
