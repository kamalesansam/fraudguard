"""Render README figures from pipeline outputs.

uv run --with matplotlib python scripts/make_figures.py
(requires `make pipeline` and `make drift-demo` first)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS, OUT = Path("reports"), Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)

metrics = json.loads((REPORTS / "metrics.json").read_text())
curve = json.loads((REPORTS / "pr_curve.json").read_text())
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot([p["recall"] for p in curve], [p["precision"] for p in curve], lw=2)
ax.scatter(
    [metrics["recall"]],
    [metrics["precision"]],
    color="crimson",
    zorder=3,
    label=f"operating point (t={metrics['threshold']:.3f})",
)
ax.set(
    xlabel="recall",
    ylabel="precision",
    xlim=(0, 1.02),
    ylim=(0, 1.02),
    title=f"Test PR curve  (PR-AUC {metrics['pr_auc']}, ROC-AUC {metrics['roc_auc']})",
)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "pr_curve.png", dpi=150)

batches = {"stable": "Stable week", "drifted": "Covariate shift", "concept": "New fraud pattern"}
reports = {k: json.loads((REPORTS / f"drift_{k}.json").read_text()) for k in batches}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
features = list(reports["drifted"]["features"])
x = range(len(features))
w = 0.27
for i, (k, label) in enumerate(batches.items()):
    a1.bar(
        [j + (i - 1) * w for j in x],
        [reports[k]["features"][f]["psi"] for f in features],
        w,
        label=label,
    )
a1.axhline(0.2, color="crimson", ls="--", lw=1, label="PSI alert (0.2)")
a1.set_xticks(list(x), features, rotation=45, ha="right", fontsize=8)
a1.set(ylabel="PSI vs training data", title="Data drift per feature")
a1.legend(fontsize=8)
recalls = [reports[k]["performance"]["current_recall"] for k in batches]
baseline = reports["stable"]["performance"]["baseline_recall"]
bars = a2.bar(list(batches.values()), recalls, color=["#4c72b0", "#dd8452", "#c44e52"])
a2.axhline(baseline, color="k", ls=":", lw=1, label=f"release recall {baseline}")
a2.bar_label(bars, fmt="%.2f")
a2.set(
    ylim=(0, 1.05),
    ylabel="fraud recall at production threshold",
    title="Concept drift: inputs look normal, recall collapses",
)
a2.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "drift.png", dpi=150)
print("wrote", sorted(p.name for p in OUT.iterdir()))
