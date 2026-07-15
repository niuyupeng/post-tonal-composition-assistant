import csv
import hashlib
import json
import math

import pytest

from post_tonal.analyze_multiseed_training import aggregate_multiseed_training, main


def _write_run(tmp_path, seed, val_losses, batch_size=4, accumulation_steps=2):
    run_dir = tmp_path / f"seed_{seed}"
    run_dir.mkdir()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch, val_loss in enumerate(val_losses, start=1):
            writer.writerow({"epoch": epoch, "train_loss": val_loss + 0.1, "val_loss": val_loss})
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "epochs_ran": len(val_losses),
                "best_val_loss": min(val_losses),
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "history": [{"not": "used for best-epoch selection"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.pt").write_bytes(f"checkpoint-{seed}".encode("ascii"))
    return run_dir


def _write_evaluation(tmp_path, seed, token_accuracy, model_loss):
    path = tmp_path / f"evaluation_seed_{seed}.json"
    path.write_text(
        json.dumps(
            {
                "experiment": f"multiseed_seed{seed}",
                "split": "test",
                "num_samples": 2000,
                "token_accuracy": token_accuracy,
                "model_loss": model_loss,
                "pcset_coverage": 0.99,
                "interval_vector_distance": 0.01,
                "range_violation_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_writes_deterministic_per_seed_and_aggregate_outputs(tmp_path, capsys):
    run_42 = _write_run(tmp_path, 42, [0.9, 0.5, 0.7], batch_size=8, accumulation_steps=2)
    run_43 = _write_run(tmp_path, 43, [0.6, 0.3], batch_size=4, accumulation_steps=4)
    evaluation_42 = _write_evaluation(tmp_path, 42, token_accuracy=0.8, model_loss=0.4)
    evaluation_43 = _write_evaluation(tmp_path, 43, token_accuracy=0.6, model_loss=0.8)
    metrics_csv = tmp_path / "outputs" / "metrics.csv"
    summary_json = tmp_path / "outputs" / "summary.json"
    latex_table = tmp_path / "outputs" / "table.tex"

    exit_code = main(
        [
            "--run-dir",
            str(run_43),
            "--run-dir",
            str(run_42),
            "--evaluation-json",
            str(evaluation_43),
            "--evaluation-json",
            str(evaluation_42),
            "--metrics-csv",
            str(metrics_csv),
            "--summary-json",
            str(summary_json),
            "--latex-table",
            str(latex_table),
        ]
    )

    assert exit_code == 0
    assert "Aggregated 2 seeds: [42, 43]" in capsys.readouterr().out
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["seed"] for row in rows] == ["42", "43"]
    assert rows[0]["epochs_ran"] == "3"
    assert rows[0]["best_epoch"] == "2"
    assert float(rows[0]["best_val_loss"]) == pytest.approx(0.5)
    assert float(rows[0]["test_token_accuracy"]) == pytest.approx(0.8)
    assert float(rows[0]["test_model_loss"]) == pytest.approx(0.4)
    assert rows[0]["test_num_samples"] == "2000"
    assert rows[0]["effective_batch_size"] == "16"
    expected_sha = hashlib.sha256(b"checkpoint-42").hexdigest()
    assert rows[0]["checkpoint_sha256"] == expected_sha
    assert rows[0]["checkpoint_path"] == (run_42 / "checkpoint.pt").as_posix()
    assert rows[0]["evaluation_path"] == evaluation_42.as_posix()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["n"] == 2
    assert summary["seeds"] == [42, 43]
    assert summary["sample_standard_deviation_ddof"] == 1
    assert summary["test_num_samples_per_seed"] == 2000
    assert summary["aggregate"]["best_val_loss"]["mean"] == pytest.approx(0.4)
    assert summary["aggregate"]["best_val_loss"]["sample_std"] == pytest.approx(
        math.sqrt(0.02)
    )
    assert summary["aggregate"]["test_token_accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["aggregate"]["test_model_loss"]["sample_std"] == pytest.approx(
        math.sqrt(0.08)
    )

    combined_outputs = metrics_csv.read_text(encoding="utf-8") + summary_json.read_text(
        encoding="utf-8"
    ) + latex_table.read_text(encoding="utf-8")
    assert "pcset_coverage" not in combined_outputs
    assert "interval_vector_distance" not in combined_outputs
    assert "range_violation_rate" not in combined_outputs
    table = latex_table.read_text(encoding="utf-8")
    assert "42 & 3 & 2 & 0.5000 & 0.8000 & 0.4000 & 16" in table
    assert "Mean $\\pm$ SD" in table
    assert "0.4000 $\\pm$ 0.1414" in table


def test_single_seed_sample_standard_deviation_is_null(tmp_path):
    run_dir = _write_run(tmp_path, 7, [1.0, 0.75])
    evaluation = _write_evaluation(tmp_path, 7, token_accuracy=0.5, model_loss=0.75)

    summary = aggregate_multiseed_training(
        [run_dir],
        [evaluation],
        tmp_path / "metrics-out.csv",
        tmp_path / "summary-out.json",
        tmp_path / "table-out.tex",
    )

    assert summary["n"] == 1
    assert all(item["sample_std"] is None for item in summary["aggregate"].values())


def test_rejects_misaligned_argument_counts_before_writing_outputs(tmp_path):
    with pytest.raises(ValueError, match="same number of times"):
        aggregate_multiseed_training(
            [tmp_path / "seed_42", tmp_path / "seed_43"],
            [tmp_path / "evaluation_seed_42.json"],
            tmp_path / "metrics-out.csv",
            tmp_path / "summary-out.json",
            tmp_path / "table-out.tex",
        )
    assert not (tmp_path / "metrics-out.csv").exists()


def test_rejects_seed_misalignment_and_missing_run_artifacts(tmp_path):
    run_dir = _write_run(tmp_path, 42, [0.8, 0.4])
    wrong_evaluation = _write_evaluation(tmp_path, 43, token_accuracy=0.7, model_loss=0.5)
    outputs = (
        tmp_path / "metrics-out.csv",
        tmp_path / "summary-out.json",
        tmp_path / "table-out.tex",
    )

    with pytest.raises(ValueError, match="Misaligned inputs"):
        aggregate_multiseed_training([run_dir], [wrong_evaluation], *outputs)
    assert not any(path.exists() for path in outputs)

    (run_dir / "metrics.csv").unlink()
    aligned_evaluation = _write_evaluation(tmp_path, 42, token_accuracy=0.7, model_loss=0.5)
    with pytest.raises(FileNotFoundError, match="Missing training metrics CSV"):
        aggregate_multiseed_training([run_dir], [aligned_evaluation], *outputs)
