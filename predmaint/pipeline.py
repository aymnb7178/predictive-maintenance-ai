"""End-to-end experiment: simulate -> validate -> features -> unit-level split -> train -> evaluate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer

from .anomaly import IForestDetector, cusum
from .config import FeatureConfig, ModelConfig, SimConfig, SplitConfig
from .evaluation import (
    alarm_summary, alarm_table, choose_threshold, clf_metrics, cost_curve, cv_rul_mae,
    fit_estimators, naive_rul, rul_metrics, split_units,
)
from .features import LoadAdjuster, build_features, feature_columns
from .model import MaintenanceModel
from .simulator import simulate_fleet
from .validation import assert_valid

log = logging.getLogger(__name__)


@dataclass
class Experiment:
    model: MaintenanceModel
    data: pd.DataFrame
    scored: pd.DataFrame  # per-row predictions joined with labels
    splits: dict[str, list[int]]
    metrics: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    importances: dict[str, pd.DataFrame] = field(default_factory=dict)


def _iforest_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if c.endswith(("_rmean", "_rstd")) or c.startswith("vib_spec")]


def run_experiment(
    sim: SimConfig | None = None,
    fcfg: FeatureConfig | None = None,
    mcfg: ModelConfig | None = None,
    scfg: SplitConfig | None = None,
    data: pd.DataFrame | None = None,
    with_cv: bool = True,
    with_permutation: bool = True,
) -> Experiment:
    """Run the full leakage-safe experiment and return the model, predictions and metrics."""
    sim, fcfg, mcfg, scfg = sim or SimConfig(), fcfg or FeatureConfig(), mcfg or ModelConfig(), scfg or SplitConfig()
    df = simulate_fleet(sim) if data is None else data.copy()
    assert_valid(df, require_labels=True)
    train_u, val_u, test_u = split_units(df["unit"].unique(), scfg)
    assert not (set(train_u) & set(val_u) | set(train_u) & set(test_u) | set(val_u) & set(test_u))
    log.info("units: train=%d val=%d test=%d", len(train_u), len(val_u), len(test_u))

    adj = LoadAdjuster.fit(df[df["unit"].isin(train_u)], fcfg)
    feats = build_features(df, adj, fcfg)
    cols = feature_columns(feats)
    full = feats.join(df[["rul", "fault_mode"]])
    full["rul_cap"] = full["rul"].clip(upper=fcfg.rul_cap)
    full["y"] = (full["rul"] <= mcfg.horizon).astype(int)
    part = {k: full[full["unit"].isin(v)] for k, v in zip(("train", "val", "test"), (train_u, val_u, test_u))}

    tr = part["train"]
    reg, clf = fit_estimators(tr[cols], tr["rul_cap"], tr["y"], mcfg)

    # Alarm threshold: validation units only.
    val_risk = clf.predict_proba(part["val"][cols])[:, 1]
    thr = choose_threshold(part["val"]["y"], val_risk, mcfg)

    # IsolationForest trained on healthy early-life rows of training units only.
    B = fcfg.baseline_cycles
    healthy = tr[(tr["cycle"] > B) & (tr["cycle"] <= 2 * B)]
    iforest = IForestDetector.fit(healthy, _iforest_columns(cols), seed=mcfg.seed)

    # CUSUM h: smallest h with no validation unit alarming before `healthy_cycles`.
    val_hi = {u: g.sort_values("cycle") for u, g in part["val"].groupby("unit")}
    h_chosen = 20.0
    for h in np.arange(1.0, 20.5, 0.5):
        bad = 0
        for g in val_hi.values():
            _, i = cusum(g["hi"].to_numpy() / adj.hi_std, mcfg.cusum_k, h)
            bad += int(i is not None and g["cycle"].iloc[i] <= mcfg.healthy_cycles)
        if bad == 0:
            h_chosen = float(h)
            break

    model = MaintenanceModel(
        feature_cfg=fcfg, model_cfg=mcfg, adjuster=adj, columns=cols, reg=reg, clf=clf,
        iforest=iforest, threshold=thr, cusum_h=h_chosen,
        meta={"units": {"train": train_u, "val": val_u, "test": test_u}, "synthetic": True,
              "train_mean_life": float(df.groupby("unit")["cycle"].max().loc[train_u].mean())},
    )
    scored = model.score_features(feats).join(full[["rul", "rul_cap", "y", "fault_mode"]])
    sp = {k: scored[scored["unit"].isin(v)] for k, v in zip(("train", "val", "test"), (train_u, val_u, test_u))}
    te = sp["test"]

    # ---- metrics (test units only; threshold from validation units) ----
    m: dict = {"n_units": {"train": len(train_u), "val": len(val_u), "test": len(test_u)}}
    naive = naive_rul(te["cycle"], model.meta["train_mean_life"], fcfg.rul_cap)
    near = te["rul"] <= 2 * mcfg.horizon
    m["rul"] = {
        "model": rul_metrics(te["rul_cap"], te["rul_pred"]),
        "naive_age_baseline": rul_metrics(te["rul_cap"], naive),
        "model_near_failure": rul_metrics(te.loc[near, "rul_cap"], te.loc[near, "rul_pred"]),
        "naive_near_failure": rul_metrics(te.loc[near, "rul_cap"], naive[near.to_numpy()]),
    }
    if with_cv:
        pool = full[full["unit"].isin(train_u + val_u)]
        maes = cv_rul_mae(pool[cols], pool["rul_cap"], pool["unit"], mcfg, scfg.cv_splits)
        m["rul"]["groupkfold_mae"] = {"folds": maes, "mean": float(np.mean(maes)), "std": float(np.std(maes))}
    m["classification"] = clf_metrics(te["y"], te["risk"], thr)
    m["classification"]["age_baseline_pr_auc"] = float(average_precision_score(te["y"], -naive))
    m["classification"]["at_0.5"] = clf_metrics(te["y"], te["risk"], 0.5)
    # Cost analysis on test rows at thresholds fixed beforehand.
    f1_thr = choose_threshold(part["val"]["y"], val_risk, ModelConfig(threshold_strategy="f1"))
    cost_rows = []
    for name, t in (("cost-optimal (val)", thr), ("F1-optimal (val)", f1_thr), ("default 0.5", 0.5)):
        c = cost_curve(te["y"], te["risk"], mcfg.c_false_alarm, mcfg.c_missed, np.array([t])).iloc[0]
        cost_rows.append({"policy": name, "threshold": t, "false_alarm_rows": int(c.fp), "missed_rows": int(c.fn), "cost": float(c.cost)})
    cost_rows.append({"policy": "never alarm", "threshold": 1.01, "false_alarm_rows": 0,
                      "missed_rows": int(te["y"].sum()), "cost": float(mcfg.c_missed * te["y"].sum())})
    m["cost_policies"] = cost_rows

    tabs = {
        "risk": alarm_table(te, "risk", thr, mcfg.persistence, mcfg.early_limit),
        "iforest": alarm_table(te, "anomaly", iforest.threshold, 3, mcfg.early_limit),
        "cusum": alarm_table(te, "cusum", h_chosen, 1, mcfg.early_limit),
    }
    m["alarms"] = {k: alarm_summary(t) for k, t in tabs.items()}
    m["alarms"]["risk_by_fault_mode"] = {
        fm: alarm_summary(t) for fm, t in tabs["risk"].groupby("fault_mode")
    }
    m["cusum_h"] = h_chosen
    m["threshold"] = thr

    imps: dict[str, pd.DataFrame] = {}
    imps["rul_impurity"] = pd.DataFrame({"feature": cols, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
    imps["clf_impurity"] = pd.DataFrame({"feature": cols, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    if with_permutation:
        v = part["val"]
        pi = permutation_importance(clf, v[cols], v["y"], scoring=make_scorer(average_precision_score, response_method="predict_proba"),
                                    n_repeats=3, random_state=0, n_jobs=1)
        imps["clf_permutation_val"] = pd.DataFrame({"feature": cols, "importance": pi.importances_mean}).sort_values("importance", ascending=False)

    return Experiment(model, df, scored, {"train": train_u, "val": val_u, "test": test_u}, m, tabs, imps)
