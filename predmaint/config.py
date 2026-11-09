"""Configuration dataclasses and logging helper."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

SENSORS: tuple[str, ...] = ("temperature", "vibration", "pressure", "current", "rpm")
# Sensors that carry degradation signal once the operating point (rpm) is removed.
HEALTH_SENSORS: tuple[str, ...] = ("temperature", "vibration", "pressure", "current")
LABEL_COLUMNS: tuple[str, ...] = ("rul", "fault_mode")


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the synthetic run-to-failure simulator."""

    n_units: int = 60
    seed: int = 42
    mean_life: float = 220.0
    life_sd: float = 45.0
    min_life: int = 120
    sudden_fault_prob: float = 0.15
    rpm_setpoints: tuple[float, ...] = (1200.0, 1800.0, 2400.0)
    regime_dwell: tuple[int, int] = (30, 80)
    noise_scale: float = 1.0
    dropout_prob: float = 0.01
    dropout_burst_prob: float = 0.003
    dropout_burst_len: tuple[int, int] = (3, 8)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureConfig:
    """Windowed feature engineering parameters."""

    window: int = 15
    ewma_span: int = 10
    spectral_window: int = 32
    baseline_cycles: int = 20  # first cycles of each unit define its own healthy baseline
    rul_cap: int = 120

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Model, labelling and alarm-policy parameters."""

    model_type: str = "gb"  # "gb" (GradientBoosting) or "rf" (RandomForest)
    horizon: int = 30  # label = failure within `horizon` cycles
    n_estimators: int = 150
    max_depth: int = 3
    learning_rate: float = 0.05
    seed: int = 0
    persistence: int = 2  # consecutive cycles above threshold to raise an alarm
    threshold_strategy: str = "cost"  # "cost" or "f1"
    c_false_alarm: float = 1.0  # cost of one false-alarm cycle (illustrative units)
    c_missed: float = 10.0  # cost of one missed at-risk cycle (illustrative units)
    watch_ratio: float = 0.5  # watch tier = watch_ratio * alarm threshold
    critical_risk: float = 0.85
    critical_rul: float = 10.0
    cusum_k: float = 0.5  # CUSUM reference value (in healthy standard deviations)
    healthy_cycles: int = 30  # CUSUM h is chosen so no validation unit alarms before this cycle
    early_limit: int = 90  # alarms earlier than this many cycles before failure count as premature

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitConfig:
    """Unit-level split. Rows of one machine never straddle two partitions."""

    val_frac: float = 0.2
    test_frac: float = 0.2
    seed: int = 7
    cv_splits: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
