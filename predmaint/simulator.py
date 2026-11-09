"""Physics-inspired synthetic run-to-failure simulator.

Each machine unit runs through operating regimes (rpm set-points). Sensors respond to
load (rpm) and to a latent wear variable ``d(t)`` that grows from 0 to 1 over the unit's
life. A fraction of units are *sudden-fault* units: they wear slowly, then a fault appears
only 6-18 cycles before failure. Sensors can drop out (single NaNs and bursts).

Everything here is invented for demonstration; it is NOT calibrated to any real machine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import SENSORS, SimConfig

log = logging.getLogger(__name__)


def _regime_path(life: int, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    regime = np.zeros(life, dtype=int)
    i, prev = 0, -1
    while i < life:
        r = int(rng.integers(len(cfg.rpm_setpoints)))
        if r == prev:
            r = (r + 1) % len(cfg.rpm_setpoints)
        dwell = int(rng.integers(cfg.regime_dwell[0], cfg.regime_dwell[1] + 1))
        regime[i : i + dwell] = r
        i, prev = i + dwell, r
    return regime


def _simulate_unit(unit: int, cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    life = int(max(cfg.min_life, round(rng.normal(cfg.mean_life, cfg.life_sd))))
    sudden = bool(rng.random() < cfg.sudden_fault_prob)
    t = np.arange(1, life + 1, dtype=float)
    ns = cfg.noise_scale

    onset = rng.uniform(0.15, 0.45) * life
    power = rng.uniform(1.8, 3.0)
    wear = np.clip((t - onset) / (life - onset), 0.0, 1.0) ** power
    if sudden:
        wear = wear * 0.3
        fault_len = int(rng.integers(6, 19))
        since = t - (life - fault_len)
        ramp = np.clip((since + 1) / 3.0, 0.0, 1.0) * (since >= 0)
        severity = ramp * (1.0 + np.clip(since, 0, None) / fault_len)
    else:
        severity = np.zeros(life)

    regime = _regime_path(life, cfg, rng)
    rpm_true = np.asarray(cfg.rpm_setpoints)[regime] + rng.normal(0, 15, life)
    load = rpm_true / 1800.0

    off_t, off_v, off_p, off_c = rng.normal(0, [1.0, 0.08, 0.05, 0.15])
    phase = rng.uniform(0, 2 * np.pi)

    # Thermal inertia: temperature relaxes toward its steady state.
    t_ss = 55 + 12 * load + 28 * wear + 8 * severity + off_t
    temp = np.empty(life)
    temp[0] = t_ss[0]
    for i in range(1, life):
        temp[i] = temp[i - 1] + 0.3 * (t_ss[i] - temp[i - 1])
    temp += rng.normal(0, 1.2 * ns, life)

    tone = np.sin(2 * np.pi * 0.31 * t + phase)
    vib = (
        1.2 + 0.8 * load**2 + 4.0 * wear**1.5 + 3.5 * severity + off_v
        + rng.normal(0, 0.15 * ns, life) * (1 + 2 * wear + severity)
        + (0.8 * wear + 0.6 * severity) * tone
    )
    pres = 5.0 + 0.5 * load - 1.2 * wear - 0.5 * severity + off_p + rng.normal(0, 0.05 * ns, life)
    cur = 8.0 + 5.0 * load + 3.5 * wear + 2.5 * severity + off_c + rng.normal(0, 0.2 * ns, life)

    df = pd.DataFrame(
        {
            "unit": unit,
            "cycle": np.arange(1, life + 1),
            "regime": regime,
            "temperature": temp,
            "vibration": vib,
            "pressure": pres,
            "current": cur,
            "rpm": rpm_true,
        }
    )
    # Sensor dropout: isolated NaNs plus short bursts on a single sensor.
    for s in SENSORS:
        mask = rng.random(life) < cfg.dropout_prob
        df.loc[mask, s] = np.nan
    for i in np.flatnonzero(rng.random(life) < cfg.dropout_burst_prob):
        s = SENSORS[int(rng.integers(len(SENSORS)))]
        n = int(rng.integers(cfg.dropout_burst_len[0], cfg.dropout_burst_len[1] + 1))
        df.loc[df.index[i : i + n], s] = np.nan

    df["rul"] = life - df["cycle"]
    df["fault_mode"] = "sudden" if sudden else "gradual"
    return df


def simulate_fleet(cfg: SimConfig | None = None) -> pd.DataFrame:
    """Simulate a fleet of run-to-failure units. Deterministic for a given ``cfg.seed``.

    Returns a long DataFrame: unit, cycle, regime, 5 sensors, rul, fault_mode.
    ``rul`` and ``fault_mode`` are labels and are never used as model features.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    frames = [_simulate_unit(u + 1, cfg, rng) for u in range(cfg.n_units)]
    df = pd.concat(frames, ignore_index=True)
    log.info(
        "simulated %d units, %d rows, %d sudden-fault units",
        cfg.n_units, len(df), df.loc[df.fault_mode == "sudden", "unit"].nunique(),
    )
    return df
