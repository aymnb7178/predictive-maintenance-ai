"""Run the full experiment on synthetic data and write charts + a sample HTML report into docs/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from predmaint.cli import _headline  # noqa: E402
from predmaint.config import setup_logging  # noqa: E402
from predmaint.evaluation import cost_curve  # noqa: E402
from predmaint.pipeline import run_experiment  # noqa: E402
from predmaint.reporting import (  # noqa: E402
    plot_cost, plot_importance, plot_lead, plot_machine, plot_pr, plot_rul, plot_sensors, save_fig, write_report,
)

DOCS = ROOT / "docs"


def main() -> None:
    setup_logging("WARNING")
    exp = run_experiment()
    m, mcfg = exp.metrics, exp.model.model_cfg
    print(_headline(m))
    print("\nCost policies on held-out units (cost = 1 x false-alarm row + 10 x missed row):")
    for r in m["cost_policies"]:
        print(f"  {r['policy']:<20} thr {r['threshold']:.2f}  FP {r['false_alarm_rows']:>3}  FN {r['missed_rows']:>3}  cost {r['cost']:.0f}")
    print("\nFirst-alarm detectors on held-out units (lead = cycles before failure):")
    for k in ("risk", "iforest", "cusum"):
        a = m["alarms"][k]
        print(f"  {k:<8} ok {a['ok']:>2}  premature {a['premature']:>2}  missed {a['missed']:>2}  median lead {a['median_lead']:.0f}")
    g = m["rul"]["groupkfold_mae"]
    print(f"\nGroupKFold (by unit) RUL MAE on train+val units: {g['mean']:.2f} +/- {g['std']:.2f}")

    te = exp.scored[exp.scored["unit"].isin(exp.splits["test"])]
    first_u = int(exp.data["unit"].iloc[0])
    save_fig(plot_sensors(exp.data[exp.data["unit"] == first_u]), DOCS / "sensors.png")
    shown = exp.tables["risk"].sort_values("unit")
    pick = [int(shown[shown.fault_mode == "gradual"].unit.iloc[0])]
    sudden = shown[shown.fault_mode == "sudden"]
    pick += [int(sudden.unit.iloc[0])] if len(sudden) else [int(shown.unit.iloc[1])]
    pick.append(int(shown.unit.iloc[-2]))
    save_fig(plot_rul(te, pick), DOCS / "rul_prediction.png")
    cc = cost_curve(te["y"], te["risk"], mcfg.c_false_alarm, mcfg.c_missed)
    save_fig(plot_cost(cc, exp.model.threshold), DOCS / "cost_threshold.png")
    save_fig(plot_pr(te["y"], te["risk"], -np.clip(exp.model.meta["train_mean_life"] - te["cycle"], 0, None)), DOCS / "pr_curve.png")
    save_fig(plot_lead({"risk model": exp.tables["risk"], "IsolationForest": exp.tables["iforest"], "CUSUM": exp.tables["cusum"]}, mcfg.early_limit), DOCS / "alarm_lead.png")
    save_fig(plot_importance(exp.importances["clf_permutation_val"], "Permutation importance (PR-AUC drop, validation units)"), DOCS / "feature_importance.png")
    g0 = te[te["unit"] == pick[0]].sort_values("cycle")
    save_fig(plot_machine(g0, exp.model.threshold, exp.model.cusum_h, None), DOCS / "machine_report.png")

    write_report(te, exp.model, DOCS / "sample_report.html", units=pick,
                 title="Sample maintenance report (synthetic held-out machines)")
    print("\ntop permutation importances:", ", ".join(exp.importances["clf_permutation_val"].head(5)["feature"]))
    (DOCS / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    print(f"wrote PNGs, sample_report.html and metrics.json to {DOCS}")


if __name__ == "__main__":
    main()
