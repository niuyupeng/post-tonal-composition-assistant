import json
from pathlib import Path

from post_tonal.full_run import (
    EXPECTED_EXPERIMENTS,
    REQUIRED_METRIC_COLUMNS,
    _metric_row_errors,
    _generation_example_errors,
    _manifest_matches_package,
    _read_command_log,
    _read_json_object,
    _read_run_incidents,
    promote_generation_examples,
    write_report,
)


def test_read_command_log_supports_utf8(tmp_path):
    log = tmp_path / "utf8.log"
    log.write_text(
        "COMMAND python -m post_tonal.train --config config.yaml\noutput\n",
        encoding="utf-8",
    )

    assert _read_command_log(log) == [
        "python -m post_tonal.train --config config.yaml"
    ]


def test_read_command_log_supports_windows_powershell_utf16(tmp_path):
    log = tmp_path / "powershell.log"
    log.write_text(
        "COMMAND python -m post_tonal.evaluate --split test\noutput\n",
        encoding="utf-16",
    )

    assert _read_command_log(log) == [
        "python -m post_tonal.evaluate --split test"
    ]


def test_read_json_object_supports_windows_powershell_utf8_bom(tmp_path):
    summary = tmp_path / "train_summary.json"
    summary.write_text('{"completed": true}', encoding="utf-8-sig")

    assert _read_json_object(summary) == {"completed": True}


def test_report_records_recovered_run_incident(tmp_path):
    incidents = tmp_path / "incidents.json"
    incidents.write_text(
        """
        {
          "incidents": [{
            "stage": "training startup",
            "status": "recovered",
            "failure": "CUDA process exited.",
            "recovery": "Health check passed and resume completed.",
            "evidence": "event-id-1000"
          }]
        }
        """,
        encoding="utf-8",
    )
    parsed, errors = _read_run_incidents(incidents)
    assert errors == []
    assert parsed[0]["status"] == "recovered"

    report = tmp_path / "report.md"
    write_report(
        output=str(report),
        metrics=str(tmp_path / "missing_metrics.csv"),
        split_summary=str(tmp_path / "missing_split.json"),
        expert_dir=str(tmp_path / "missing_expert"),
        constraints=str(tmp_path / "missing_constraints.csv"),
        examples=str(tmp_path / "missing_examples.json"),
        run_root=str(tmp_path / "missing_runs"),
        log_path=str(tmp_path / "missing.log"),
        main_table=str(tmp_path / "missing_main.tex"),
        ablation_table=str(tmp_path / "missing_ablation.tex"),
        incidents=str(incidents),
    )

    text = report.read_text(encoding="utf-8")
    assert "## Failed or Retried Stages" in text
    assert "training startup [recovered]" in text
    assert "event-id-1000" in text


def test_promote_generation_examples_rewrites_only_versioned_artifact_paths(
    tmp_path,
):
    source = tmp_path / "v3_examples.json"
    output = tmp_path / "canonical_examples.json"
    source.write_text(
        """
        [
          {
            "experiment": "proposed_constraint_guided_transformer",
            "musicxml": "results\\\\eval_musicxml_v3\\\\proposed\\\\one.musicxml",
            "analysis_report": "results\\\\eval_musicxml_v3\\\\proposed\\\\one.json"
          },
          {
            "experiment": "rule_baseline",
            "musicxml": "results\\\\eval_musicxml\\\\rule\\\\one.musicxml",
            "analysis_report": "results\\\\eval_musicxml\\\\rule\\\\one.json"
          }
        ]
        """,
        encoding="utf-8",
    )

    promote_generation_examples(str(source), str(output))

    promoted = json.loads(output.read_text(encoding="utf-8"))
    assert promoted[0]["musicxml"] == "results/eval_musicxml/proposed/one.musicxml"
    assert promoted[0]["analysis_report"] == "results/eval_musicxml/proposed/one.json"
    assert promoted[1]["musicxml"] == "results\\eval_musicxml\\rule\\one.musicxml"
    assert promoted[1]["analysis_report"] == "results\\eval_musicxml\\rule\\one.json"


def test_generation_example_validation_rejects_missing_artifacts(tmp_path):
    examples = tmp_path / "examples.json"
    examples.write_text(
        """
        [{
          "experiment": "rule_baseline",
          "musicxml": "missing.musicxml",
          "analysis_report": "missing.json"
        }]
        """,
        encoding="utf-8",
    )

    errors = _generation_example_errors(examples)

    assert "generation example 0: missing missing.musicxml" in errors
    assert "generation example 0: missing missing.json" in errors
    assert "rule_baseline: expected 20 generation examples, found 1" in errors


def test_manifest_validation_rejects_paths_from_another_package(tmp_path):
    expert_root = tmp_path / "expert_eval" / "project2"
    xml_path = expert_root / "musicxml" / "project2_01.musicxml"
    manifest = {
        "count": 20,
        "items": [
            {
                "id": f"project2_{index:02d}",
                "musicxml": str(
                    tmp_path
                    / "expert_eval"
                    / "project2_v3"
                    / "musicxml"
                    / f"project2_{index:02d}.musicxml"
                ),
                "analysis_report": str(
                    tmp_path
                    / "expert_eval"
                    / "project2_v3"
                    / "analysis_reports"
                    / f"project2_{index:02d}.json"
                ),
            }
            for index in range(1, 21)
        ],
    }
    xml_paths = [
        xml_path.with_name(f"project2_{index:02d}.musicxml")
        for index in range(1, 21)
    ]

    assert not _manifest_matches_package(manifest, xml_paths, expert_root)


def test_incomplete_report_does_not_infer_experiment_failure(tmp_path):
    report = tmp_path / "report.md"

    write_report(
        output=str(report),
        metrics=str(tmp_path / "missing_metrics.csv"),
        split_summary=str(tmp_path / "missing_split.json"),
        expert_dir=str(tmp_path / "missing_expert"),
        constraints=str(tmp_path / "missing_constraints.csv"),
        examples=str(tmp_path / "missing_examples.json"),
        run_root=str(tmp_path / "missing_runs"),
        log_path=str(tmp_path / "missing.log"),
        main_table=str(tmp_path / "missing_main.tex"),
        ablation_table=str(tmp_path / "missing_ablation.tex"),
    )

    text = report.read_text(encoding="utf-8")
    assert "## Pending Evaluation Rows" in text
    assert "## Missing or Incomplete Neural Checkpoints" in text
    assert "## Neural Checkpoint Details" in text
    assert "No failed or retried stage was recorded" in text


def test_metric_row_validation_rejects_non_test_and_missing_values():
    rows = []
    for experiment in EXPECTED_EXPERIMENTS:
        row = {
            "experiment": experiment,
            "split": "test",
            "num_samples": "2000",
            **{field: "0.5" for field in REQUIRED_METRIC_COLUMNS},
        }
        rows.append(row)

    assert _metric_row_errors(rows) == []
    rows[0]["split"] = "val"
    rows[1]["gesture_consistency_score"] = ""

    errors = _metric_row_errors(rows)
    assert "rule_baseline: split must be test" in errors
    assert (
        "vanilla_transformer: gesture_consistency_score is missing or non-finite"
        in errors
    )


def test_report_records_malformed_musicxml_instead_of_crashing(tmp_path):
    expert = tmp_path / "expert"
    musicxml = expert / "musicxml"
    reports = expert / "analysis_reports"
    musicxml.mkdir(parents=True)
    reports.mkdir()
    (musicxml / "project2_01.musicxml").write_text("<score-partwise>", encoding="utf-8")
    report = tmp_path / "report.md"

    write_report(
        output=str(report),
        metrics=str(tmp_path / "missing_metrics.csv"),
        split_summary=str(tmp_path / "missing_split.json"),
        expert_dir=str(expert),
        constraints=str(tmp_path / "missing_constraints.csv"),
        examples=str(tmp_path / "missing_examples.json"),
        run_root=str(tmp_path / "missing_runs"),
        log_path=str(tmp_path / "missing.log"),
        main_table=str(tmp_path / "missing_main.tex"),
        ablation_table=str(tmp_path / "missing_ablation.tex"),
    )

    text = report.read_text(encoding="utf-8")
    assert "Invalid MusicXML/report artifact" in text


def test_windows_runner_passes_reused_checkpoint_to_evaluation():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_project2_full_local.ps1"
    ).read_text(encoding="utf-8")

    assert "$usesCheckpoint = $exp.Train -or $exp.ContainsKey(\"ReuseFrom\")" in script
    assert "if ($usesCheckpoint) {" in script
    assert "[System.Text.UTF8Encoding]::new($false)" in script
    assert '--incidents", "results/project2_v3_run_incidents.json"' in script
    assert (
        '"results\\project2_v3_constraint_summary.svg") '
        '-Destination (Join-Path $Root "results\\project2_constraint_summary.svg")'
        in script
    )
    assert '"post_tonal.full_run", "promote-generation-examples"' in script
    assert '"--output-dir", "expert_eval/project2"' in script
    assert '-Filter "*_test_*.musicxml"' in script
