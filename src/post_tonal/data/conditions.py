"""Condition-prefix transformations for controlled corpus ablations."""

from __future__ import annotations

import copy
from typing import Any


EVALUATION_ONLY_METADATA_KEYS = {
    "target_density_curve",
}


def apply_condition_ablation(metadata: dict[str, Any], ablation: str | None) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    for key in EVALUATION_ONLY_METADATA_KEYS:
        result.pop(key, None)
    if ablation in {"no_constraints", "serial_only", "rhythm_only", "gesture_only", "no_pcset"}:
        result["pcset"] = []
        result["interval_vector"] = None
    if ablation in {"no_constraints", "pcset_only", "rhythm_only", "gesture_only", "no_serial"}:
        result["row"] = None
        result["row_form"] = None
    if ablation in {"no_constraints", "pcset_only", "serial_only", "gesture_only", "no_rhythm"}:
        result["rhythm_profile"] = None
    if ablation in {"no_constraints", "pcset_only", "serial_only", "rhythm_only", "no_gesture"}:
        result["gesture"] = None
    return result
