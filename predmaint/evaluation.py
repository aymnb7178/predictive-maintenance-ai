"""Leakage-safe evaluation: unit-level splits, baselines, alarm lead time and cost analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from .config import ModelConfig, SplitConfig

log = logging.getLogger(__name__)


def split_units(units, cfg: SplitConfig | None = None) -> tuple[list[int], list[int], list[int]]:
    """Seeded unit-level train / validation / test split. Rows of one unit stay together."""
    cfg = cfg or SplitConfig()
    u = np.array(sorted(set(int(x) for x in units)))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(u)
    n_test, n_val = int(round(len(u) * cfg.test_frac)), int(round(len(u) * cfg.val_frac))
    test, val, train = u[:n_test], u[n_test : n_test + n_val], u[n_test + n_val :]
    return sorted(train.tolist()), sorted(val.tolist()), sorted(test.tolist())


def make_estimators(cfg: ModelConfig):
    """Return (RUL regressor, failure-within-horizon classifier)."""
    if cfg.model_type == "rf":
        reg = RandomForestRegressor(n_estimators=cfg.n_estimators, min_samples_leaf=5, random_state=cfg.seed, n_jobs=1)
        clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=cfg.seed, n_jobs=1,
        )
    elif cfg.model_type == "gb":
        kw = dict(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
                  subsample=0.8, random_state=cfg.seed)
        reg, clf = GradientBoostingRegressor(**kw), GradientBoostingClassifier(**kw)
    else:
        raise ValueError(f"unknown model_type {cfg.model_type!r}")
    return reg, clf


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-row weights that give both classes equal total weight (imbalance handling)."""
    y = np.asarray(y).astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def fit_estimators(X, rul_capped, y, cfg: ModelConfig):
    reg, clf = make_estimators(cfg)
    reg.fit(X, rul_capped)
    if cfg.model_type == "gb":
        clf.fit(X, y, sample_weight=balanced_weights(y))
    else:
        clf.fit(X, y)
    return reg, clf


def naive_rul(cycle: np.ndarray, train_mean_life: float, cap: int) -> np.ndarray:
    """Baseline: expected remaining life = mean training life - age, clipped to [0, cap]."""
    return np.clip(train_mean_life - np.asarray(cycle, dtype=float), 0, cap)


def rul_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def clf_metrics(y, risk, threshold: float) -> dict[str, float]:
    pred = (np.asarray(risk) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, risk)),
        "prevalence": float(np.mean(y)),
    }


def cost_curve(y, risk, c_false_alarm: float, c_missed: float, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Total cost vs. threshold: ``c_false_alarm * FP + c_missed * FN`` (per at-risk row)."""
    y, risk = np.asarray(y).astype(int), np.asarray(risk)
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    rows = []
    for t in grid:
        p = risk >= t
        fp, fn = int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
        rows.append({"threshold": float(t), "fp": fp, "fn": fn, "cost": c_false_alarm * fp + c_missed * fn})
    return pd.DataFrame(rows)


def choose_threshold(y, risk, cfg: ModelConfig) -> float:
    """Pick the alarm threshold. Call ONLY with validation-unit data."""
    grid = np.linspace(0.02, 0.98, 49)
    if cfg.threshold_strategy == "cost":
        cc = cost_curve(y, risk, cfg.c_false_alarm, cfg.c_missed, grid)
        return float(cc.loc[cc["cost"].idxmin(), "threshold"])
    if cfg.threshold_strategy == "f1":
        f1s = [f1_score(y, (np.asarray(risk) >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(f1s))])
    raise ValueError(f"unknown threshold_strategy {cfg.threshold_strategy!r}")


def first_alarm(score: np.ndarray, threshold: float, persistence: int = 1) -> int | None:
    """Index where ``score >= threshold`` has held for ``persistence`` consecutive cycles."""
    run = 0
    for i, v in enumerate(np.asarray(score, dtype=float)):
        run = run + 1 if v >= threshold else 0
        if run >= persistence:
            return i
    return None


def alarm_table(frame: pd.DataFrame, score_col: str, threshold: float, persistence: int, early_limit: int) -> pd.DataFrame:
    """Per-unit first alarm. ``lead`` = cycles before failure (= rul at the alarm cycle).

    status: ``missed`` (no alarm), ``premature`` (lead > early_limit), ``ok`` otherwise.
    """
    rows = []
    for u, g in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        i = first_alarm(g[score_col].to_numpy(), threshold, persistence)
        if i is None:
            rows.append({"unit": u, "alarm_cycle": np.nan, "lead": np.nan, "status": "missed"})
            continue
        lead = int(g["rul"].iloc[i])
        rows.append({"unit": u, "alarm_cycle": int(g["cycle"].iloc[i]), "lead": lead,
                     "status": "premature" if lead > early_limit else "ok"})
    tab = pd.DataFrame(rows)
    if "fault_mode" in frame:
        tab = tab.merge(frame.groupby("unit")["fault_mode"].first().rename("fault_mode"), on="unit")
    return tab


def alarm_summary(tab: pd.DataFrame) -> dict[str, float]:
    ok = tab[tab["status"] == "ok"]
    out = {
        "n_units": int(len(tab)),
        "ok": int(len(ok)),
        "missed": int((tab["status"] == "missed").sum()),
        "premature": int((tab["status"] == "premature").sum()),
        "median_lead": float(ok["lead"].median()) if len(ok) else float("nan"),
        "min_lead": float(ok["lead"].min()) if len(ok) else float("nan"),
        "max_lead": float(ok["lead"].max()) if len(ok) else float("nan"),
    }
    return out


def cv_rul_mae(X: pd.DataFrame, rul_capped, groups, cfg: ModelConfig, n_splits: int) -> list[float]:
    """GroupKFold (by unit) RUL MAE over a set of units; folds never share a machine."""
    maes = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, rul_capped, groups):
        reg, _ = make_estimators(cfg)
        reg.fit(X.iloc[tr], np.asarray(rul_capped)[tr])
        maes.append(float(mean_absolute_error(np.asarray(rul_capped)[te], reg.predict(X.iloc[te]))))
    return maes
