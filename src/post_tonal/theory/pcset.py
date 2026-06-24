"""Pitch-class set utilities for post-tonal symbolic analysis."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


def normalize_pcset(pcs: Iterable[int]) -> list[int]:
    """Return sorted unique pitch classes modulo 12."""

    return sorted({int(pc) % 12 for pc in pcs})


def transpose(pcs: Iterable[int], n: int) -> list[int]:
    """Transpose a pitch-class set by Tn."""

    return normalize_pcset((int(pc) + int(n)) % 12 for pc in pcs)


def invert(pcs: Iterable[int], n: int = 0) -> list[int]:
    """Invert a pitch-class set by TnI, mapping pc to n - pc modulo 12."""

    return normalize_pcset((int(n) - int(pc)) % 12 for pc in pcs)


def _rotations(pcs: list[int]) -> list[list[int]]:
    rotations: list[list[int]] = []
    for i in range(len(pcs)):
        rot = pcs[i:] + [x + 12 for x in pcs[:i]]
        rotations.append(rot)
    return rotations


def _normal_order_unmodded(pcs: Iterable[int]) -> list[int]:
    ordered = normalize_pcset(pcs)
    if len(ordered) <= 1:
        return ordered

    candidates = _rotations(ordered)

    def key(seq: list[int]) -> tuple[int, tuple[int, ...], int]:
        span = seq[-1] - seq[0]
        # Tie-break by packing from the right, then by starting pc for determinism.
        right_packing = tuple(seq[-1] - x for x in reversed(seq[:-1]))
        return (span, right_packing, seq[0] % 12)

    return min(candidates, key=key)


def normal_order(pcs: Iterable[int]) -> list[int]:
    """Compute a practical normal order for small pc sets."""

    return [pc % 12 for pc in _normal_order_unmodded(pcs)]


def _zero_based(seq: Iterable[int]) -> list[int]:
    values = list(seq)
    if not values:
        return []
    base = values[0]
    return [int(x - base) % 12 for x in values]


def prime_form(pcs: Iterable[int]) -> list[int]:
    """Return a compact prime-form approximation for a pc set.

    The implementation follows the usual normal-order versus inverted-normal-order
    comparison and is intentionally small enough for this repository's generated
    set labels.
    """

    ordered = _zero_based(_normal_order_unmodded(pcs))
    inverted = _zero_based(_normal_order_unmodded(invert(pcs)))
    return min(ordered, inverted)


def interval_vector(pcs: Iterable[int]) -> list[int]:
    """Compute the six-entry interval-class vector."""

    values = normalize_pcset(pcs)
    vector = [0, 0, 0, 0, 0, 0]
    for a, b in combinations(values, 2):
        diff = (b - a) % 12
        ic = min(diff, 12 - diff)
        if 1 <= ic <= 6:
            vector[ic - 1] += 1
    return vector


def pcset_coverage(generated_pcs: Iterable[int], target_pcs: Iterable[int]) -> float:
    """Fraction of target pitch classes appearing in the generated material."""

    target = set(normalize_pcset(target_pcs))
    if not target:
        return 1.0
    generated = set(normalize_pcset(generated_pcs))
    return len(target & generated) / len(target)


def interval_vector_distance(a: Iterable[int], b: Iterable[int]) -> float:
    """L1 distance between two pc-set interval vectors."""

    va = interval_vector(a)
    vb = interval_vector(b)
    return float(sum(abs(x - y) for x, y in zip(va, vb)))


def interval_vector_distance_from_vector(pcs: Iterable[int], target_vector: Iterable[int]) -> float:
    actual = interval_vector(pcs)
    target = [int(x) for x in target_vector]
    if len(target) != 6:
        raise ValueError("Interval vector targets must contain six entries.")
    return float(sum(abs(x - y) for x, y in zip(actual, target)))
