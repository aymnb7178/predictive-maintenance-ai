"""Command line interface: simulate, train, predict, report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import FeatureConfig, ModelConfig, SimConfig, SplitConfig, setup_logging
from .model import MaintenanceModel
from .pipeline import run_experiment
from .reporting import DISCLAIMER, machine_summary, write_report
from .simulator import simulate_fleet
from .validation import DataValidationError

log = logging.getLogger("predmaint.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predmaint", description="Predictive maintenance on SYNTHETIC sensor data (demo project).")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a synthetic run-to-failure fleet to CSV")
    s.add_argument("--units", type=int, default=60)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    s.add_argument("--strip-labels", action="store_true", help="drop rul/fault_mode (looks like new readings)")

    t = sub.add_parser("train", help="simulate, evaluate on held-out units, and save a model")
    t.add_argument("--units", type=int, default=60)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--model-type", choices=["gb", "rf"], default="gb")
    t.add_argument("--horizon", type=int, default=30, help="failure-within-N-cycles label")
    t.add_argument("--strategy", choices=["cost", "f1"], default="cost", help="how the alarm threshold is chosen (validation units)")
    t.add_argument("--c-false-alarm", type=float, default=1.0)
    t.add_argument("--c-missed", type=float, default=10.0)
    t.add_argument("--no-cv", action="store_true", help="skip GroupKFold (faster)")
    t.add_argument("--out", default="models/model.joblib")
    t.add_argument("--metrics-out", default=None, help="optional JSON file for all metrics")

    r = sub.add_parser("predict", help="score a CSV of new readings (unit, cycle, 5 sensors)")
    r.add_argument("--model", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--output", default=None, help="optional CSV with a row per input cycle")

    h = sub.add_parser("report", help="HTML maintenance report per machine")
    h.add_argument("--model", required=True)
    h.add_argument("--input", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--units", type=int, nargs="*", default=None)
    return p


def _headline(m: dict) -> str:
    rul, c, a = m["rul"], m["classification"], m["alarms"]["risk"]
    lines = [
        "Held-out units only (SYNTHETIC data; optimistic, not real-world evidence):",
        f"  units train/val/test : {m['n_units']['train']}/{m['n_units']['val']}/{m['n_units']['test']}",
        f"  RUL MAE  model {rul['model']['mae']:.1f}  vs naive age baseline {rul['naive_age_baseline']['mae']:.1f} cycles",
        f"  RUL RMSE model {rul['model']['rmse']:.1f}  vs naive age baseline {rul['naive_age_baseline']['rmse']:.1f} cycles",
        f"  failure<=N cycles    : precision {c['precision']:.2f}  recall {c['recall']:.2f}  F1 {c['f1']:.2f}  PR-AUC {c['pr_auc']:.2f} (prevalence {c['prevalence']:.2f})",
        f"  alarm threshold      : {m['threshold']:.2f} (chosen on validation units)",
        f"  first-alarm lead     : median {a['median_lead']:.0f} cycles; missed {a['missed']}, premature {a['premature']} of {a['n_units']} units",
    ]
    return "\n".join(lines)


def _load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return pd.read_csv(path)


def cmd_simulate(a) -> int:
    df = simulate_fleet(SimConfig(n_units=a.units, seed=a.seed))
    if a.strip_labels:
        df = df.drop(columns=["rul", "fault_mode", "regime"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows for {a.units} synthetic units to {a.out}")
    return 0


def cmd_train(a) -> int:
    mcfg = ModelConfig(model_type=a.model_type, horizon=a.horizon, threshold_strategy=a.strategy,
                       c_false_alarm=a.c_false_alarm, c_missed=a.c_missed)
    exp = run_experiment(SimConfig(n_units=a.units, seed=a.seed), FeatureConfig(), mcfg, SplitConfig(),
                         with_cv=not a.no_cv, with_permutation=False)
    path = exp.model.save(a.out)
    if a.metrics_out:
        Path(a.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.metrics_out).write_text(json.dumps(exp.metrics, indent=2, default=float))
    print(_headline(exp.metrics))
    print(f"model saved to {path}")
    return 0


def cmd_predict(a) -> int:
    model = MaintenanceModel.load(a.model)
    scored = model.predict(_load_csv(a.input))
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(a.output, index=False)
    rows = [machine_summary(g, model) for _, g in scored.groupby("unit")]
    out = pd.DataFrame(rows)[["unit", "last_cycle", "risk", "rul_pred", "tier", "first_alarm_cycle"]]
    out = out.rename(columns={"rul_pred": "est_rul", "last_cycle": "cycle"})
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_report(a) -> int:
    model = MaintenanceModel.load(a.model)
    scored = model.predict(_load_csv(a.input))
    path = write_report(scored, model, a.out, units=a.units)
    print(f"report written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.WARNING)
    try:
        return {"simulate": cmd_simulate, "train": cmd_train, "predict": cmd_predict, "report": cmd_report}[a.cmd](a)
    except (DataValidationError, FileNotFoundError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
