"""MusicXML export for generated symbolic fragments."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised when music21 is installed locally.
    from music21 import duration, instrument as m21instrument, metadata as m21metadata, meter, note, stream

    MUSIC21_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback is covered by tests in minimal environments.
    duration = m21instrument = m21metadata = meter = note = stream = None
    MUSIC21_AVAILABLE = False

from post_tonal.utils import ensure_dir


if MUSIC21_AVAILABLE:
    INSTRUMENT_CLASSES = {
        "piano": m21instrument.Piano,
        "flute": m21instrument.Flute,
        "clarinet": m21instrument.Clarinet,
        "violin": m21instrument.Violin,
        "cello": m21instrument.Violoncello,
        "generic_voice": m21instrument.Vocalist,
    }
else:
    INSTRUMENT_CLASSES = {}


def _instrument(name: str):
    if not MUSIC21_AVAILABLE:
        return None
    cls = INSTRUMENT_CLASSES.get(name, m21instrument.Vocalist)
    return cls()


def events_to_score(events: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> stream.Score:
    if not MUSIC21_AVAILABLE:
        raise RuntimeError("music21 is not installed; use export_musicxml for the built-in fallback writer.")
    metadata = metadata or {}
    voices = max(1, int(metadata.get("voices", max([int(e.get("voice", 0)) for e in events], default=0) + 1)))
    measures = min(16, max(1, int(metadata.get("measures", 4))))
    target_beats = measures * 4.0
    instrument_name = metadata.get("instrument", "generic_voice")
    score = stream.Score(id="post_tonal_fragment")
    score.metadata = m21metadata.Metadata()
    score.metadata.title = metadata.get("title", "Synthetic post-tonal fragment")
    score.metadata.composer = metadata.get("creator", "Synthetic post-tonal research output")

    for voice_idx in range(voices):
        part = stream.Part(id=f"voice_{voice_idx}")
        part.insert(0, _instrument(instrument_name))
        part.insert(0, meter.TimeSignature("4/4"))
        part.partName = f"{instrument_name}_{voice_idx + 1}"
        voice_events = [
            event
            for event in events
            if int(event.get("voice", 0)) == voice_idx
            and 0.0 <= float(event.get("onset", 0.0)) < target_beats
        ]
        highest_end = 0.0
        for event in voice_events:
            onset = float(event.get("onset", 0.0))
            ql = min(
                max(0.25, float(event.get("duration", 0.25))),
                target_beats - onset,
            )
            if event.get("is_rest", False) or event.get("pitch") is None:
                obj = note.Rest()
            else:
                obj = note.Note(int(event["pitch"]))
            obj.duration = duration.Duration(ql)
            part.insert(onset, obj)
            highest_end = max(highest_end, onset + ql)
        if highest_end < target_beats:
            padding = note.Rest()
            padding.duration = duration.Duration(0.25)
            padding.style.hideObjectOnPrint = True
            part.insert(target_beats - 0.25, padding)
        measured_part = part.makeMeasures(inPlace=False)
        written_measures = list(measured_part.getElementsByClass(stream.Measure))
        for extra in written_measures[measures:]:
            measured_part.remove(extra)
        while len(list(measured_part.getElementsByClass(stream.Measure))) < measures:
            measure_number = len(list(measured_part.getElementsByClass(stream.Measure))) + 1
            missing_measure = stream.Measure(number=measure_number)
            full_rest = note.Rest()
            full_rest.duration = duration.Duration(4.0)
            missing_measure.append(full_rest)
            measured_part.append(missing_measure)
        score.insert(0, measured_part)
    return score


def export_musicxml(events: list[dict[str, Any]], path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    if not MUSIC21_AVAILABLE:
        path_obj.write_text(_fallback_musicxml(events, metadata or {}), encoding="utf-8")
        return path_obj
    score = events_to_score(events, metadata)
    score.write("musicxml", fp=str(path_obj))
    return path_obj


PITCH_NAMES = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("G", 1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def _duration_ticks(quarter_length: float) -> int:
    return max(1, int(round(float(quarter_length) * 4)))


def _note_xml(event: dict[str, Any], duration_ticks: int) -> str:
    if event.get("is_rest", False) or event.get("pitch") is None:
        return f"      <note><rest/><duration>{duration_ticks}</duration><type>quarter</type></note>\n"
    midi = int(event["pitch"])
    step, alter = PITCH_NAMES[midi % 12]
    octave = midi // 12 - 1
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    return (
        "      <note>"
        f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
        f"<duration>{duration_ticks}</duration><type>quarter</type></note>\n"
    )


def _fallback_musicxml(events: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    measures = max(1, int(metadata.get("measures", 4)))
    voices = max(1, int(metadata.get("voices", max([int(e.get("voice", 0)) for e in events], default=0) + 1)))
    title = escape(str(metadata.get("title", "Synthetic post-tonal fragment")))
    instrument = escape(str(metadata.get("instrument", "generic_voice")))
    creator = escape(str(metadata.get("creator", "Synthetic post-tonal research output")))
    parts = []
    part_list = []
    for voice_idx in range(voices):
        part_id = f"P{voice_idx + 1}"
        part_list.append(
            f'    <score-part id="{part_id}"><part-name>{instrument} {voice_idx + 1}</part-name></score-part>\n'
        )
        part_lines = [f'  <part id="{part_id}">\n']
        voice_events = sorted(
            [event for event in events if int(event.get("voice", 0)) == voice_idx],
            key=lambda event: float(event.get("onset", 0.0)),
        )
        for measure_idx in range(measures):
            start = measure_idx * 4.0
            end = start + 4.0
            part_lines.append(f'    <measure number="{measure_idx + 1}">\n')
            if measure_idx == 0:
                part_lines.append(
                    "      <attributes><divisions>4</divisions><key><fifths>0</fifths></key>"
                    "<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>\n"
                )
            cursor = start
            for event in [e for e in voice_events if start <= float(e.get("onset", 0.0)) < end]:
                onset = float(event.get("onset", 0.0))
                if onset > cursor:
                    part_lines.append(_note_xml({"is_rest": True}, _duration_ticks(onset - cursor)))
                ql = min(float(event.get("duration", 0.25)), end - onset)
                part_lines.append(_note_xml(event, _duration_ticks(ql)))
                cursor = max(cursor, onset + ql)
            if cursor < end:
                part_lines.append(_note_xml({"is_rest": True}, _duration_ticks(end - cursor)))
            part_lines.append("    </measure>\n")
        part_lines.append("  </part>\n")
        parts.append("".join(part_lines))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">\n'
        f"  <work><work-title>{title}</work-title></work>\n"
        f"  <identification><creator type=\"composer\">{creator}</creator></identification>\n"
        "  <part-list>\n"
        + "".join(part_list)
        + "  </part-list>\n"
        + "".join(parts)
        + "</score-partwise>\n"
    )
