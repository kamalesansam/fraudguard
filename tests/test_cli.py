"""End-to-end run of every pipeline stage through the CLI, in a temp directory."""

import json

from typer.testing import CliRunner

from fraudguard.cli import app

runner = CliRunner()


def run(*args: str) -> int:
    result = runner.invoke(app, list(args))
    assert result.exit_code in (0, 2), result.output
    return result.exit_code


def test_full_pipeline(tmp_path):
    raw, processed, model, reports = (
        tmp_path / "raw.csv",
        tmp_path / "processed",
        tmp_path / "model",
        tmp_path / "reports",
    )
    run("generate", "--out", str(raw), "--n-rows", "20000")
    run("validate", "--data", str(raw), "--report", str(reports / "validation.json"))
    run("split", "--data", str(raw), "--out-dir", str(processed))
    run("train", "--data-dir", str(processed), "--model-dir", str(model))
    run(
        "evaluate",
        "--data-dir", str(processed),
        "--model-dir", str(model),
        "--metrics-out", str(reports / "metrics.json"),
        "--curve-out", str(reports / "pr_curve.json"),
    )  # fmt: skip
    run(
        "package",
        "--data-dir", str(processed),
        "--model-dir", str(model),
        "--metrics-file", str(reports / "metrics.json"),
        "--no-mlflow",
    )  # fmt: skip
    manifest = json.loads((model / "manifest.json").read_text())
    assert manifest["metrics"]["roc_auc"] >= 0.95
    assert (model / "MODEL_CARD.md").exists()

    concept = tmp_path / "concept.csv"
    run("generate", "--out", str(concept), "--n-rows", "8000", "--seed", "9", "--concept-drift")
    code = run(
        "drift", str(concept),
        "--reference", str(processed / "train.csv"),
        "--model-dir", str(model),
        "--report", str(reports / "drift.json"),
    )  # fmt: skip
    assert code == 2


def test_validate_blocks_bad_data(tmp_path):
    raw = tmp_path / "raw.csv"
    run("generate", "--out", str(raw), "--n-rows", "2000")
    text = raw.read_text().splitlines()
    text[1] = text[1].replace(text[1].split(",")[1], "-1.0", 1)
    raw.write_text("\n".join(text) + "\n")
    result = runner.invoke(
        app, ["validate", "--data", str(raw), "--report", str(tmp_path / "v.json")]
    )
    assert result.exit_code == 1
    assert "amount" in result.output


def test_hash_key():
    result = runner.invoke(app, ["hash-key", "--api-key", "abc"])
    assert result.output.strip() == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_export_openapi(tmp_path):
    out = tmp_path / "openapi.json"
    run("export-openapi", "--out", str(out))
    assert "/v1/predict" in json.loads(out.read_text())["paths"]
