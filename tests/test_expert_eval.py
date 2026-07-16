import csv
import json
from xml.etree import ElementTree

from post_tonal.export_musicxml import export_musicxml
from post_tonal.prepare_expert_eval import prepare_from_evaluation_examples


def test_model_expert_package_is_anonymized_and_conditioned(tmp_path):
    metadata = {
        "pcset": [0, 1, 4],
        "row": None,
        "row_form": None,
        "rhythm_profile": "sparse",
        "gesture": "fragmented",
        "voices": 2,
        "measures": 4,
        "instrument": "violin",
        "creator": "Named source system",
    }
    events = [{"voice": 0, "onset": 0.0, "duration": 1.0, "pitch": 60, "instrument": "violin"}]
    source_xml = tmp_path / "source.musicxml"
    source_report = tmp_path / "source.json"
    export_musicxml(events, source_xml, metadata)
    source_report.write_text(json.dumps({"metadata": metadata, "analysis": {"pcset_coverage": 1.0}}), encoding="utf-8")
    examples_path = tmp_path / "examples.json"
    examples_path.write_text(
        json.dumps(
            [
                {
                    "experiment": "controlled_constraint_reranked",
                    "sample_id": "sample-1",
                    "musicxml": str(source_xml),
                    "analysis_report": str(source_report),
                    "musicxml_structurally_valid": True,
                    "musicxml_measure_count_adherent": True,
                    "musicxml_voice_count_adherent": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "expert"
    result = prepare_from_evaluation_examples(examples_path, "controlled_constraint_reranked", output, 1)

    assert result["examples"] == 1
    root = ElementTree.parse(output / "musicxml" / "project2_01.musicxml").getroot()
    creator = root.find("./identification/creator")
    assert creator is not None
    assert creator.text == "Anonymous score-level composition assistant"
    assert root.find("./identification/encoding/encoding-date") is None
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run"] == "controlled_constraint_reranked"
    assert manifest["count"] == 1
    assert manifest["items"][0]["sample_id"] == "sample-1"
    form = (output / "blind_rating_form_project2.md").read_text(encoding="utf-8")
    assert "pc-set {0,1,4}" in form
    assert "row form not specified" in form
    with open(output / "blind_rating_form_project2.csv", newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["target_pcset"] == "0,1,4"
    assert csv_row["target_voices"] == "2"
    assert csv_row["target_measures"] == "4"
    assert "automatic metric values are withheld" in (output / "README.md").read_text(encoding="utf-8")
