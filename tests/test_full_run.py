from pathlib import Path

from post_tonal.full_run import (
    EXPECTED_EXPERIMENTS,
    REQUIRED_METRIC_COLUMNS,
    _metric_row_errors,
    _read_command_log,
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
    assert "No experiment failure is inferred from a missing artifact." in text


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
