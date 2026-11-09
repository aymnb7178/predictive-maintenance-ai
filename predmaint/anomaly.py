"""Anomaly detection: IsolationForest on healthy-trained features and a CUSUM control chart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


def cusum(x: np.ndarray, k: float = 0.5, h: float = 5.0) -> tuple[np.ndarray, int | None]:
    """One-sided (upper) CUSUM on a standardised signal.

    ``S_t = max(0, S_{t-1} + x_t - k)``. Returns the statistic and the first index where
    ``S_t > h`` (or None). NaNs are treated as zero deviation.
    """
    s = np.zeros(len(x))
    acc = 0.0
    first: int | None = None
    for i, v in enumerate(np.nan_to_num(np.asarray(x, dtype=float))):
        acc = max(0.0, acc + v - k)
        s[i] = acc
        if first is None and acc > h:
            first = i
    return s, first


@dataclass
class IForestDetector:
    """IsolationForest trained only on healthy rows; score 0-1 style (higher = more anomalous)."""

    columns: list[str]
    model: IsolationForest
    threshold: float = 0.0

    @classmethod
    def fit(cls, X, columns: list[str], seed: int = 0, quantile: float = 0.995) -> "IForestDetector":
        m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(X[columns])
        det = cls(columns, m)
        det.threshold = float(np.quantile(det.score(X), quantile))
        return det

    def score(self, X) -> np.ndarray:
        return -self.model.score_samples(X[self.columns])
