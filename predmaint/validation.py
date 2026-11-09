"""Input validation for sensor readings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SENSORS

PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-20.0, 250.0),
    "vibration": (0.0, 50.0),
    "pressure": (0.0, 50.0),
    "current": (0.0, 200.0),
    "rpm": (0.0, 10000.0),
}


class DataValidationError(ValueError):
    """Raised when readings cannot be used safely."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return f"{'OK' if self.ok else 'INVALID'}: {len(self.errors)} errors, {len(self.warnings)} warnings"


def validate_readings(
    df: pd.DataFrame,
    *,
    require_labels: bool = False,
    max_missing_frac: float = 0.3,
    warn_missing_frac: float = 0.05,
) -> ValidationReport:
    """Check schema, types, ordering, duplicates, missingness and physical ranges."""
    rep = ValidationReport()
    required = ["unit", "cycle", *SENSORS] + (["rul"] if require_labels else [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"missing columns: {missing_cols}")
        return rep
    if len(df) == 0:
        rep.errors.append("no rows")
        return rep

    for c in SENSORS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            rep.errors.append(f"column '{c}' is not numeric")
    if rep.errors:
        return rep

    vals = df[list(SENSORS)].to_numpy(dtype=float)
    if np.isinf(vals).any():
        rep.errors.append("infinite values in sensor columns")
    if df["unit"].isna().any() or df["cycle"].isna().any():
        rep.errors.append("null unit or cycle ids")
        return rep
    if df.duplicated(["unit", "cycle"]).any():
        rep.errors.append(f"{int(df.duplicated(['unit', 'cycle']).sum())} duplicate (unit, cycle) rows")

    gaps = 0
    for _, g in df.groupby("unit", sort=False):
        d = np.diff(g["cycle"].to_numpy())
        if (d <= 0).any():
            rep.errors.append(f"unit {g['unit'].iloc[0]}: cycles not strictly increasing")
            break
        gaps += int((d > 1).sum())
    if gaps:
        rep.warnings.append(f"{gaps} gaps in cycle numbering")

    miss = df[list(SENSORS)].isna().mean()
    rep.stats["missing_frac"] = {k: round(float(v), 4) for k, v in miss.items()}
    for s, frac in miss.items():
        if frac > max_missing_frac:
            rep.errors.append(f"sensor '{s}' is {frac:.0%} missing")
        elif frac > warn_missing_frac:
            rep.warnings.append(f"sensor '{s}' is {frac:.0%} missing")

    for s, (lo, hi) in PHYSICAL_RANGES.items():
        bad = int(((df[s] < lo) | (df[s] > hi)).sum())
        if bad:
            rep.warnings.append(f"sensor '{s}': {bad} readings outside physical range [{lo}, {hi}]")
    rep.stats["n_rows"] = int(len(df))
    rep.stats["n_units"] = int(df["unit"].nunique())
    return rep


def assert_valid(df: pd.DataFrame, **kwargs) -> ValidationReport:
    """Validate and raise :class:`DataValidationError` on any error."""
    rep = validate_readings(df, **kwargs)
    if not rep.ok:
        raise DataValidationError("; ".join(rep.errors))
    return rep
