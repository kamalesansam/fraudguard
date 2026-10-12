"""`fraudguard` command line. Each pipeline stage in dvc.yaml calls one command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from fraudguard import config

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def generate(
    out: Path = config.DATA_RAW,
    n_rows: int | None = None,
    seed: int | None = None,
    drift: float = 0.0,
    concept_drift: bool = False,
) -> None:
    """Generate the seeded synthetic transaction dataset."""
    from fraudguard.data.generate import generate_transactions

    p = config.load_params()
    df = generate_transactions(
        n_rows=n_rows or p["generate"]["n_rows"],
        fraud_rate=p["generate"]["fraud_rate_target"],
        seed=p["seed"] if seed is None else seed,
        drift=drift,
        concept_drift=concept_drift,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"wrote {len(df)} rows ({df['is_fraud'].mean():.2%} fraud) -> {out}")


@app.command()
def validate(
    data: Path = config.DATA_RAW,
    report: Path = config.REPORTS / "validation.json",
) -> None:
    """Enforce the data contract. Non-zero exit blocks the pipeline."""
    from fraudguard.data.validate import validate_transactions, write_report

    p = config.load_params()["validate"]
    result = validate_transactions(
        pd.read_csv(data),
        fraud_rate_min=p["fraud_rate_min"],
        fraud_rate_max=p["fraud_rate_max"],
        max_duplicate_share=p["max_duplicate_share"],
    )
    write_report(result, report)
    if not result.passed:
        _fail("data validation failed:\n  " + "\n  ".join(result.errors))
    typer.echo(f"data contract passed: {result.n_rows} rows, fraud rate {result.fraud_rate:.4f}")


@app.command()
def split(data: Path = config.DATA_RAW, out_dir: Path = config.DATA_PROCESSED) -> None:
    """Stratified train / validation / test split."""
    p = config.load_params()
    df = pd.read_csv(data)
    s = p["split"]
    train_val, test = train_test_split(
        df, test_size=s["test_size"], stratify=df["is_fraud"], random_state=p["seed"]
    )
    val_share = s["val_size"] / (1 - s["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["is_fraud"], random_state=p["seed"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out_dir / f"{name}.csv", index=False)
        typer.echo(f"{name}: {len(part)} rows, {part['is_fraud'].mean():.2%} fraud")


@app.command()
def train(data_dir: Path = config.DATA_PROCESSED, model_dir: Path = config.MODEL_DIR) -> None:
    """Fit the pipeline and save it with skops (no pickle)."""
    from fraudguard.integrity import save_model
    from fraudguard.train import fit

    p = config.load_params()
    model = fit(pd.read_csv(data_dir / "train.csv"), p["train"], p["seed"])
    digest = save_model(model, model_dir / "model.skops")
    typer.echo(f"saved model.skops sha256={digest}")


@app.command()
def evaluate(
    data_dir: Path = config.DATA_PROCESSED,
    model_dir: Path = config.MODEL_DIR,
    metrics_out: Path = config.REPORTS / "metrics.json",
    curve_out: Path = config.REPORTS / "pr_curve.json",
) -> None:
    """Pick the threshold on validation, score test, enforce the quality gate."""
    from fraudguard.evaluate import (
        SLICE_COLUMN,
        check_gate,
        compute_metrics,
        pick_threshold,
        pr_curve_points,
        predict_scores,
    )
    from fraudguard.integrity import load_verified, sha256_file

    p = config.load_params()["evaluate"]
    model_path = model_dir / "model.skops"
    model = load_verified(model_path, sha256_file(model_path))
    val, test = pd.read_csv(data_dir / "val.csv"), pd.read_csv(data_dir / "test.csv")

    threshold = pick_threshold(
        val["is_fraud"].to_numpy(), predict_scores(model, val), p["target_precision"]
    )
    y_test, test_scores = test["is_fraud"].to_numpy(), predict_scores(model, test)
    metrics = compute_metrics(y_test, test_scores, threshold, test[SLICE_COLUMN])
    gate = check_gate(metrics, p["gate"])
    metrics["gate"] = {"passed": gate.passed, "failures": gate.failures}

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    curve_out.write_text(json.dumps(pr_curve_points(y_test, test_scores), indent=2) + "\n")
    typer.echo(json.dumps({k: metrics[k] for k in ("roc_auc", "pr_auc", "precision", "recall")}))
    if not gate.passed:
        _fail("quality gate failed:\n  " + "\n  ".join(gate.failures))
    typer.echo("quality gate passed")


@app.command()
def package(
    data_dir: Path = config.DATA_PROCESSED,
    model_dir: Path = config.MODEL_DIR,
    metrics_file: Path = config.REPORTS / "metrics.json",
    mlflow: bool = True,
) -> None:
    """Write manifest.json + MODEL_CARD.md and log the run to MLflow."""
    from fraudguard.package import build_manifest, log_to_mlflow, write_package

    metrics = json.loads(metrics_file.read_text())
    if not metrics["gate"]["passed"]:
        _fail("refusing to package a model that failed the quality gate")
    manifest = build_manifest(
        model_dir / "model.skops",
        data_dir / "train.csv",
        {k: v for k, v in metrics.items() if k != "gate"},
        config.load_params(),
        config.ROOT,
    )
    write_package(model_dir, manifest)
    run_id = log_to_mlflow(model_dir, manifest) if mlflow else None
    typer.echo(f"packaged model sha256={manifest['model']['sha256']} mlflow_run={run_id}")


@app.command()
def drift(
    current: Annotated[Path, typer.Argument(help="Production batch CSV")],
    reference: Path = config.DATA_PROCESSED / "train.csv",
    model_dir: Path = config.MODEL_DIR,
    report: Path = config.REPORTS / "drift.json",
    html: Path | None = None,
) -> None:
    """Compare a production batch against training data. Exit 2 on drift alert."""
    from fraudguard.integrity import load_verified
    from fraudguard.monitor.drift import run_drift_check, write_evidently_html

    p = config.load_params()["drift"]
    ref_df, cur_df = pd.read_csv(reference), pd.read_csv(current)
    model = manifest = None
    if (model_dir / "manifest.json").exists():
        manifest = json.loads((model_dir / "manifest.json").read_text())
        model = load_verified(model_dir / manifest["model"]["file"], manifest["model"]["sha256"])

    result = run_drift_check(ref_df, cur_df, p, model=model, manifest=manifest)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    if html is not None and write_evidently_html(ref_df, cur_df, html):
        typer.echo(f"evidently report -> {html}")

    drifted = [k for k, v in result["features"].items() if v["drifted"]]
    typer.echo(f"drifted features: {drifted or 'none'} (share {result['drifted_share']:.0%})")
    if "performance" in result:
        perf = result["performance"]
        typer.echo(f"recall {perf['baseline_recall']} -> {perf['current_recall']}")
    if result["alert"]:
        typer.secho("DRIFT ALERT: retraining required", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)
    typer.echo("no actionable drift")


@app.command("hash-key")
def hash_key(api_key: str = typer.Option(..., prompt=True, hide_input=True)) -> None:
    """Print the SHA-256 of an API key for FRAUDGUARD_API_KEY_HASHES (store hashes, not keys)."""
    typer.echo(hashlib.sha256(api_key.encode()).hexdigest())


@app.command("export-openapi")
def export_openapi(out: Path = Path("openapi.json")) -> None:
    """Write the API's OpenAPI schema (used by the ZAP DAST scan)."""
    from fraudguard.serve.app import create_app

    out.write_text(json.dumps(create_app(load_model=False).openapi(), indent=2) + "\n")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
