from pathlib import Path
from xml.etree import ElementTree

from post_tonal.export_musicxml import export_musicxml
from post_tonal.evaluate import musicxml_structurally_valid
from post_tonal.models.rule_generator import RuleGenerator
from post_tonal.theory.pcset import interval_vector


def test_musicxml_export(tmp_path: Path):
    metadata = {
        "pcset": [0, 1, 4, 6],
        "interval_vector": interval_vector([0, 1, 4, 6]),
        "rhythm_profile": "medium",
        "gesture": "fragmented",
        "voices": 2,
        "measures": 4,
        "instrument": "generic_voice",
    }
    events = RuleGenerator(seed=5).generate(metadata)
    out = export_musicxml(events, tmp_path / "fragment.musicxml", metadata)
    assert out.exists()
    assert out.stat().st_size > 0
    root = ElementTree.parse(out).getroot()
    assert root.tag.endswith("score-partwise")
    assert root.findtext("./identification/creator") == "Synthetic post-tonal research output"
    parts = root.findall("./part")
    assert len(parts) == metadata["voices"]
    assert all(len(part.findall("./measure")) == metadata["measures"] for part in parts)
    assert musicxml_structurally_valid(out)


def test_musicxml_export_pads_requested_span(tmp_path: Path):
    metadata = {
        "pcset": [0, 1, 4],
        "interval_vector": interval_vector([0, 1, 4]),
        "rhythm_profile": "sparse",
        "gesture": "silence_gap",
        "voices": 3,
        "measures": 8,
        "instrument": "clarinet",
    }
    events = [
        {
            "onset": 0.0,
            "duration": 0.25,
            "voice": 0,
            "pitch": 60,
            "pc": 0,
            "is_rest": False,
            "instrument": "clarinet",
        }
    ]
    out = export_musicxml(events, tmp_path / "padded.musicxml", metadata)
    root = ElementTree.parse(out).getroot()
    parts = root.findall("./part")
    assert len(parts) == 3
    assert all(len(part.findall("./measure")) == 8 for part in parts)
