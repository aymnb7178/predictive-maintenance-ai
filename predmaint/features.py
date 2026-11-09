"""Causal windowed feature engineering.

Every feature at cycle ``t`` uses only readings from cycles ``<= t`` of the same unit
(verified by a truncation test). Sensors are first load-adjusted: the dependence on rpm
is modelled on healthy early-life data of the *training* units, and each unit's own
baseline (first ``baseline_cycles`` cycles, expanding so it stays causal) is subtracted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import HEALTH_SENSORS, SENSORS, FeatureConfig

log = logging.getLogger(__name__)

DEGRADE_SIGN: dict[str, float] = {"temperature": 1.0, "vibration": 1.0, "pressure": -1.0, "current": 1.0}
TONE_BIN_RANGE = (9, 11)  # FFT bins around 0.31 cycles/sample for a 32-sample window
ID_COLUMNS = ("unit", "cycle")


@dataclass
class LoadAdjuster:
    """Fitted on healthy early-life rows of training units only."""

    coefs: dict[str, np.ndarray] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    fill: dict[str, float] = field(default_factory=dict)
    hi_std: float = 1.0

    @classmethod
    def fit(cls, df: pd.DataFrame, cfg: FeatureConfig) -> "LoadAdjuster":
        """Fit load curves (quadratic in rpm/1800) and healthy residual scales."""
        adj = cls()
        early = df[df["cycle"] <= 2 * cfg.baseline_cycles].copy()
        adj.fill = {s: float(df[s].median()) for s in SENSORS}
        early["load"] = early["rpm"].fillna(adj.fill["rpm"]) / 1800.0
        for s in HEALTH_SENSORS:
            sub = early[["load", s]].dropna()
            adj.coefs[s] = np.polyfit(sub["load"], sub[s], 2)
            adj.scale[s] = 1.0
        # Residual scale after removing each unit's own baseline (cycles B+1..2B).
        zs = []
        for s in HEALTH_SENSORS:
            parts = []
            for _, g in early.groupby("unit"):
                res = g[s] - np.polyval(adj.coefs[s], g["load"])
                base = np.nanmedian(res.iloc[: cfg.baseline_cycles])
                parts.append((res.iloc[cfg.baseline_cycles :] - base).to_numpy())
            r = np.concatenate(parts)
            adj.scale[s] = float(np.nanstd(r)) or 1.0
            zs.append(DEGRADE_SIGN[s] * r / adj.scale[s])
        adj.hi_std = float(np.nanstd(np.mean(zs, axis=0))) or 1.0
        return adj


def _ewma(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _rolling_slope(x: pd.Series, t: pd.Series, w: int) -> pd.Series:
    """Least-squares slope over a trailing window, vectorised via rolling moments."""
    mt, mx = t.rolling(w, min_periods=3).mean(), x.rolling(w, min_periods=3).mean()
    mtx = (t * x).rolling(w, min_periods=3).mean()
    vt = (t * t).rolling(w, min_periods=3).mean() - mt**2
    return ((mtx - mt * mx) / vt.replace(0, np.nan)).fillna(0.0)


def spectral_features(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Trailing-window FFT energy proxies (front-padded with the first value, so causal)."""
    pad = np.concatenate([np.full(window - 1, x[0]), x])
    win = sliding_window_view(pad, window)
    win = win - win.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(win * np.hanning(window), axis=1)) ** 2 / window
    total = spec[:, 1:].sum(axis=1)
    lo, hi = TONE_BIN_RANGE
    band = spec[:, lo : hi + 1].sum(axis=1)
    return {
        "vib_spec_energy": np.log1p(total),
        "vib_spec_band": np.log1p(band),
        "vib_spec_ratio": band / (total + 1e-9),
    }


def _unit_features(g: pd.DataFrame, adj: LoadAdjuster, cfg: FeatureConfig) -> pd.DataFrame:
    x = g[list(SENSORS)].ffill()
    for s in SENSORS:
        x[s] = x[s].fillna(adj.fill[s])
    load = x["rpm"].to_numpy() / 1800.0
    t = pd.Series(g["cycle"].to_numpy(dtype=float), index=g.index)
    out: dict[str, np.ndarray | pd.Series] = {"unit": g["unit"].to_numpy(), "cycle": g["cycle"].to_numpy()}
    out["age"] = t.to_numpy()
    out["rpm"] = x["rpm"].to_numpy()
    zs = []
    B, w = cfg.baseline_cycles, cfg.window
    for s in HEALTH_SENSORS:
        res = pd.Series(x[s].to_numpy() - np.polyval(adj.coefs[s], load), index=g.index)
        head = res.iloc[:B].expanding().median()  # causal baseline
        base = pd.concat([head, pd.Series(head.iloc[-1], index=res.index[len(head) :])])
        z = DEGRADE_SIGN[s] * (res - base) / adj.scale[s]
        zs.append(z)
        out[f"{s}_z"] = z.to_numpy()
        out[f"{s}_rmean"] = z.rolling(w, min_periods=3).mean().fillna(0.0).to_numpy()
        out[f"{s}_rstd"] = z.rolling(w, min_periods=3).std().fillna(0.0).to_numpy()
        out[f"{s}_slope"] = _rolling_slope(z, t, w).to_numpy()
        out[f"{s}_ewma"] = _ewma(z, cfg.ewma_span).to_numpy()
    hi = pd.concat(zs, axis=1).mean(axis=1)
    out["hi"] = hi.to_numpy()
    out["hi_rmean"] = hi.rolling(w, min_periods=3).mean().fillna(0.0).to_numpy()
    out["hi_slope"] = _rolling_slope(hi, t, w).to_numpy()
    out.update(spectral_features(zs[HEALTH_SENSORS.index("vibration")].to_numpy(), cfg.spectral_window))
    return pd.DataFrame(out, index=g.index)


def build_features(df: pd.DataFrame, adj: LoadAdjuster, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute causal features per unit. Rows keep the order and index of ``df``.

    Labels (``rul``, ``fault_mode``) are never read here.
    """
    cfg = cfg or FeatureConfig()
    parts = [
        _unit_features(g.sort_values("cycle"), adj, cfg) for _, g in df.groupby("unit", sort=False)
    ]
    return pd.concat(parts).loc[df.index]


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns: everything except identifiers and the raw composite ``hi``."""
    return [c for c in feats.columns if c not in ID_COLUMNS and c != "hi"]
