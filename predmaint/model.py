"""Trained maintenance model bundle: features, estimators, detectors, thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .anomaly import IForestDetector, cusum
from .config import FeatureConfig, ModelConfig
from .features import LoadAdjuster, build_features
from .validation import assert_valid

log = logging.getLogger(__name__)

TIERS = ("normal", "watch", "plan_maintenance", "critical")
TIER_ACTIONS = {
    "normal": "No action. Continue routine monitoring.",
    "watch": "Increase inspection frequency; review sensor trends.",
    "plan_maintenance": "Schedule maintenance in the next window; order parts.",
    "critical": "Stop or derate the machine and inspect immediately.",
}


def action_tier(risk: float, rul: float, threshold: float, cfg: ModelConfig) -> str:
    """Map risk / estimated RUL to an action tier."""
    if risk >= cfg.critical_risk or rul <= cfg.critical_rul:
        return "critical"
    if risk >= threshold:
        return "plan_maintenance"
    if risk >= cfg.watch_ratio * threshold:
        return "watch"
    return "normal"


@dataclass
class MaintenanceModel:
    """Everything needed to score new readings. Persisted with joblib."""

    feature_cfg: FeatureConfig
    model_cfg: ModelConfig
    adjuster: LoadAdjuster
    columns: list[str]
    reg: object
    clf: object
    iforest: IForestDetector
    threshold: float
    cusum_h: float
    meta: dict = field(default_factory=dict)

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.adjuster, self.feature_cfg)

    def score_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Risk, RUL estimate, anomaly scores and tier for each feature row."""
        X = feats[self.columns]
        out = feats[["unit", "cycle"]].copy()
        out["risk"] = self.clf.predict_proba(X)[:, 1]
        out["rul_pred"] = np.clip(self.reg.predict(X), 0, self.feature_cfg.rul_cap)
        out["anomaly"] = self.iforest.score(feats)
        out["cusum"] = 0.0
        for u, idx in feats.groupby("unit").groups.items():
            order = feats.loc[idx].sort_values("cycle").index
            s, _ = cusum(feats.loc[order, "hi"].to_numpy() / self.adjuster.hi_std, self.model_cfg.cusum_k, self.cusum_h)
            out.loc[order, "cusum"] = s
        out["health_index"] = 100.0 * np.clip(out["rul_pred"] / self.feature_cfg.rul_cap, 0, 1)
        out["tier"] = [
            action_tier(r, u, self.threshold, self.model_cfg) for r, u in zip(out["risk"], out["rul_pred"])
        ]
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and score raw readings (needs each unit's history from its first cycle)."""
        assert_valid(df)
        return self.score_features(self.featurize(df))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "MaintenanceModel":
        obj = joblib.load(path)
        if not isinstance(obj, MaintenanceModel):
            raise TypeError("file does not contain a MaintenanceModel")
        return obj
