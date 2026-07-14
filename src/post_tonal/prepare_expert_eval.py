"""Prepare a blind expert-evaluation package for Project 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

from post_tonal.export_musicxml import export_musicxml
from post_tonal.models.rule_generator import DEFAULT_PCSETS, RuleGenerator
from post_tonal.theory.analysis_report import analyze_events
from post_tonal.theory.gesture import GESTURE_LABELS
from post_tonal.theory.pcset import interval_vector
from post_tonal.theory.rhythm_profile import RHYTHM_PROFILES
from post_tonal.theory.serial import generate_twelve_tone_row
from post_tonal.utils import INSTRUMENTS, ensure_dir, save_json


RATING_DIMENSIONS = [
    "post-tonal material coherence",
    "pc-set / interval consistency",
    "serial logic clarity",
    "rhythmic-profile control",
    "gesture recognizability",
    "notational usefulness",
    "usefulness for contemporary composition sketching",
]

EXPERT_METRICS = [
    "pcset_coverage",
    "interval_vector_distance",
    "row_order_accuracy",
    "aggregate_completion_rate",
    "rhythmic_profile_distance",
    "gesture_consistency_score",
    "range_violation_rate",
]


def _metadata(idx: int) -> dict:
    pcset = DEFAULT_PCSETS[idx % len(DEFAULT_PCSETS)]
    row = generate_twelve_tone_row(seed=1000 + idx) if idx % 2 == 0 else None
    forms = ["P0", "R0", "I0", "RI0", "P5", "I7"]
    return {
        "pcset": pcset,
        "interval_vector": interval_vector(pcset),
        "row": row,
        "row_form": forms[idx % len(forms)] if row else None,
        "rhythm_profile": RHYTHM_PROFILES[idx % len(RHYTHM_PROFILES)],
        "gesture": GESTURE_LABELS[idx % len(GESTURE_LABELS)],
        "voices": 2 + (idx % 5),
        "measures": 4 + (idx % 5),
        "instrument": INSTRUMENTS[idx % len(INSTRUMENTS)],
    }


def _condition_value(value: object) -> str:
    if value is None:
        return "not specified"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def write_forms(root: Path, items: list[dict]) -> None:
    md_lines = [
        "# Blind Rating Form: Project 2",
        "",
        "Rater ID: __________",
        "",
        "Relevant expertise and years: ________________________________________________",
        "",
        "Use a 1-7 scale for each dimension: 1 = very weak, 4 = adequate, and 7 = very strong.",
        "Enter NA for serial logic when no row form is specified. Rate only the symbolic score sketch against the stated conditions.",
        "The system identity is withheld; do not infer or rate authorship, style imitation, or audio quality.",
        "",
    ]
    for item in items:
        metadata = item["condition_metadata"]
        md_lines.append(f"## {item['id']}")
        md_lines.append(
            "- target conditions: "
            f"pc-set {{{_condition_value(metadata.get('pcset'))}}}; "
            f"row form {_condition_value(metadata.get('row_form'))}; "
            f"rhythm {_condition_value(metadata.get('rhythm_profile'))}; "
            f"gesture {_condition_value(metadata.get('gesture'))}; "
            f"voices {_condition_value(metadata.get('voices'))}; "
            f"measures {_condition_value(metadata.get('measures'))}"
        )
        for dimension in RATING_DIMENSIONS:
            md_lines.append(f"- {dimension}: ___ / 7")
        md_lines.append("- comments: ")
        md_lines.append("")
    (root / "blind_rating_form_project2.md").write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = root / "blind_rating_form_project2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        condition_fields = [
            "target_pcset",
            "target_row_form",
            "target_rhythm_profile",
            "target_gesture",
            "target_voices",
            "target_measures",
        ]
        rating_fields = [dimension.replace(" ", "_").replace("/", "slash") for dimension in RATING_DIMENSIONS]
        fieldnames = ["rater_id", "rater_expertise", "example_id"] + condition_fields + rating_fields + ["comments"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            metadata = item["condition_metadata"]
            row = {
                "rater_id": "",
                "rater_expertise": "",
                "example_id": item["id"],
                "target_pcset": _condition_value(metadata.get("pcset")),
                "target_row_form": _condition_value(metadata.get("row_form")),
                "target_rhythm_profile": _condition_value(metadata.get("rhythm_profile")),
                "target_gesture": _condition_value(metadata.get("gesture")),
                "target_voices": _condition_value(metadata.get("voices")),
                "target_measures": _condition_value(metadata.get("measures")),
            }
            for field in rating_fields + ["comments"]:
                row[field] = ""
            writer.writerow(row)


def write_package_readme(root: Path, items: list[dict]) -> None:
    content = f"""# Project 2 Expert Evaluation Package

This blinded package contains {len(items)} score-level MusicXML examples generated by one fixed score-generation configuration. Generator identity and automatic metric values are withheld from raters.

## Materials for raters

- `musicxml/`: anonymized MusicXML scores.
- `blind_rating_form_project2.md`: printable form with target conditions.
- `blind_rating_form_project2.csv`: one machine-readable template per rater.

Use the 1-7 anchors in the form. Enter `NA` for serial logic when no row form is specified. Return one separately named CSV per rater without changing `example_id` values.

## Materials withheld from raters

- `manifest.json` records provenance and file mappings for the research team.
- `analysis_reports/` contains automatic reports that could bias human judgments.

Before collecting identifiable expert data, obtain any institutional ethics determination or consent required by the investigators' institution. No ratings are included in this repository until they are genuinely collected.
"""
    (root / "README.md").write_text(content, encoding="utf-8")


def _anonymize_musicxml(source: Path, destination: Path) -> None:
    tree = ElementTree.parse(source)
    root = tree.getroot()
    identification = root.find("identification")
    if identification is None:
        identification = ElementTree.SubElement(root, "identification")
    creators = identification.findall("creator")
    creator = creators[0] if creators else ElementTree.SubElement(identification, "creator", {"type": "composer"})
    creator.set("type", "composer")
    creator.text = "Anonymous score-level composition assistant"
    for extra_creator in creators[1:]:
        identification.remove(extra_creator)
    encoding = identification.find("encoding")
    if encoding is not None:
        for date_element in list(encoding.findall("encoding-date")):
            encoding.remove(date_element)
    ElementTree.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def prepare_from_evaluation_examples(
    examples_json: str | Path,
    experiment: str,
    output_dir: str | Path,
    count: int,
) -> dict[str, int | str]:
    examples = json.loads(Path(examples_json).read_text(encoding="utf-8"))
    selected = [
        item
        for item in examples
        if item.get("experiment") == experiment and item.get("musicxml_structurally_valid", False)
    ][:count]
    if len(selected) != count:
        raise ValueError(f"Expected {count} structurally valid {experiment!r} examples, found {len(selected)}.")

    root = ensure_dir(output_dir)
    examples_dir = ensure_dir(root / "musicxml")
    reports_dir = ensure_dir(root / "analysis_reports")
    manifest: list[dict] = []
    for idx, example in enumerate(selected, start=1):
        example_id = f"project2_{idx:02d}"
        source_xml = Path(example["musicxml"])
        source_report = Path(example["analysis_report"])
        if not source_xml.exists() or not source_report.exists():
            raise FileNotFoundError(f"Missing evaluation artifact for {example_id}: {source_xml} or {source_report}")
        report_payload = json.loads(source_report.read_text(encoding="utf-8"))
        metadata = report_payload.get("metadata", {})
        analysis = report_payload.get("analysis", example.get("analysis", {}))
        musicxml_path = examples_dir / f"{example_id}.musicxml"
        report_path = reports_dir / f"{example_id}.json"
        _anonymize_musicxml(source_xml, musicxml_path)
        save_json({"metadata": metadata, "analysis": analysis}, report_path)
        manifest.append(
            {
                "id": example_id,
                "musicxml": str(musicxml_path),
                "analysis_report": str(report_path),
                "condition_metadata": {
                    key: metadata.get(key)
                    for key in ["pcset", "row", "row_form", "rhythm_profile", "gesture", "voices", "measures"]
                },
                "metrics": {key: analysis.get(key) for key in EXPERT_METRICS},
                "source": experiment,
                "sample_id": example.get("sample_id"),
            }
        )
    save_json({"run": experiment, "count": count, "items": manifest}, root / "manifest.json")
    write_forms(root, manifest)
    write_package_readme(root, manifest)
    return {"examples": count, "source": experiment}


def prepare_expert_eval(output_dir: str | Path = "expert_eval/project2", count: int = 20, seed: int = 2026) -> dict[str, int]:
    root = ensure_dir(output_dir)
    examples_dir = ensure_dir(root / "musicxml")
    reports_dir = ensure_dir(root / "analysis_reports")
    manifest = []
    for idx in range(count):
        metadata = _metadata(idx)
        events = RuleGenerator(seed=seed + idx).generate(metadata)
        report = analyze_events(events, metadata)
        example_id = f"project2_{idx + 1:02d}"
        musicxml_path = examples_dir / f"{example_id}.musicxml"
        report_path = reports_dir / f"{example_id}.json"
        export_musicxml(events, musicxml_path, metadata)
        save_json({"metadata": metadata, "analysis": report}, report_path)
        manifest.append({"example_id": example_id, "musicxml": str(musicxml_path), "analysis_report": str(report_path)})
    save_json(manifest, root / "manifest.json")
    form_items = [
        {
            "id": f"project2_{idx + 1:02d}",
            "condition_metadata": _metadata(idx),
        }
        for idx in range(count)
    ]
    write_forms(root, form_items)
    write_package_readme(root, form_items)
    return {"examples": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="expert_eval/project2")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--examples-json", default=None)
    parser.add_argument("--experiment", default="controlled_constraint_reranked")
    args = parser.parse_args()
    if args.examples_json:
        print(prepare_from_evaluation_examples(args.examples_json, args.experiment, args.output_dir, args.count))
    else:
        print(prepare_expert_eval(args.output_dir, args.count, args.seed))


if __name__ == "__main__":
    main()
