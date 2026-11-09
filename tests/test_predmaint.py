import math

import numpy as np
import pandas as pd
import pytest

from predmaint.anomaly import cusum
from predmaint.cli import main
from predmaint.config import FeatureConfig, ModelConfig, SENSORS, SimConfig, SplitConfig
from predmaint.evaluation import (
    alarm_table, balanced_weights, choose_threshold, cost_curve, first_alarm, naive_rul, split_units,
)
from predmaint.features import LoadAdjuster, build_features, feature_columns, spectral_features
from predmaint.model import MaintenanceModel, action_tier
from predmaint.pipeline import run_experiment
from predmaint.reporting import build_report
from predmaint.simulator import simulate_fleet
from predmaint.validation import DataValidationError, assert_valid, validate_readings

SIM = SimConfig(n_units=24, seed=3, mean_life=150, life_sd=20, sudden_fault_prob=0.3)
MCFG = ModelConfig(n_estimators=40)


@pytest.fixture(scope="module")
def fleet():
    return simulate_fleet(SIM)


@pytest.fixture(scope="module")
def exp():
    return run_experiment(SIM, FeatureConfig(), MCFG, SplitConfig(), with_cv=False, with_permutation=False)


# ---- simulator
def test_simulator_deterministic(fleet):
    pd.testing.assert_frame_equal(fleet, simulate_fleet(SIM))
    assert not fleet.equals(simulate_fleet(SimConfig(n_units=24, seed=4, mean_life=150, life_sd=20)))


def test_simulator_run_to_failure_labels(fleet):
    for _, g in fleet.groupby("unit"):
        assert g["rul"].iloc[-1] == 0 and g["rul"].is_monotonic_decreasing
        assert len(g) >= SIM.min_life


def test_simulator_has_dropout_regimes_and_sudden_faults(fleet):
    assert fleet[list(SENSORS)].isna().any().any()
    assert fleet.groupby("unit")["regime"].nunique().max() >= 2
    assert set(fleet["fault_mode"]) == {"gradual", "sudden"}


def test_simulator_degradation_visible(fleet):
    g = fleet[fleet.fault_mode == "gradual"]
    early = g[g.rul > 150].groupby("regime")["vibration"].mean()
    late = g[g.rul < 10].groupby("regime")["vibration"].mean()
    assert (late.reindex(early.index).dropna() > early.reindex(late.index).dropna()).all()


# ---- validation
def test_validation_accepts_simulated(fleet):
    rep = assert_valid(fleet, require_labels=True)
    assert rep.ok and rep.stats["n_units"] == 24


def test_validation_rejects_bad_input(fleet):
    base = fleet.head(50).copy()
    assert "missing columns" in validate_readings(base.drop(columns=["vibration"])).errors[0]
    dup = pd.concat([base, base.head(1)])
    assert any("duplicate" in e for e in validate_readings(dup).errors)
    bad = base.copy(); bad["rpm"] = bad["rpm"].astype(str)
    assert any("not numeric" in e for e in validate_readings(bad).errors)
    inf = base.copy(); inf.loc[inf.index[0], "current"] = np.inf
    assert not validate_readings(inf).ok
    assert validate_readings(base.iloc[0:0]).errors == ["no rows"]
    with pytest.raises(DataValidationError):
        assert_valid(base.drop(columns=["unit"]))


def test_validation_warns_on_range_and_gaps(fleet):
    d = fleet[fleet.unit == 1].head(40).copy()
    d.loc[d.index[3], "temperature"] = 900.0
    d = d.drop(d.index[10])
    rep = validate_readings(d)
    assert rep.ok and any("physical range" in w for w in rep.warnings) and any("gaps" in w for w in rep.warnings)


# ---- features
def test_features_are_causal(fleet):
    train = fleet[fleet.unit <= 12]
    adj = LoadAdjuster.fit(train, FeatureConfig())
    g = fleet[fleet.unit == 20]
    full = build_features(g, adj)
    cut = build_features(g[g.cycle <= 100], adj)
    cols = feature_columns(full)
    np.testing.assert_allclose(full.loc[cut.index, cols].to_numpy(), cut[cols].to_numpy(), atol=1e-9)


def test_features_exclude_labels_and_have_no_nans(fleet):
    adj = LoadAdjuster.fit(fleet[fleet.unit <= 12], FeatureConfig())
    f = build_features(fleet, adj)
    cols = feature_columns(f)
    assert not {"rul", "fault_mode", "unit", "cycle", "hi"} & set(cols)
    assert not f[cols].isna().any().any() and np.isfinite(f[cols].to_numpy()).all()
    assert {"vibration_rmean", "vibration_rstd", "vibration_slope", "vibration_ewma", "vib_spec_energy"} <= set(cols)


def test_spectral_band_responds_to_tone():
    t = np.arange(200)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 200)
    tone = noise + 3 * np.sin(2 * np.pi * 0.31 * t)
    a, b = spectral_features(noise, 32), spectral_features(tone, 32)
    assert b["vib_spec_ratio"][40:].mean() > 2 * a["vib_spec_ratio"][40:].mean()


# ---- anomaly
def test_cusum_detects_shift_not_noise():
    rng = np.random.default_rng(1)
    flat = rng.normal(0, 1, 300)
    _, i0 = cusum(flat, k=0.5, h=8)
    shifted = np.concatenate([flat[:100], rng.normal(2, 1, 100)])
    s, i1 = cusum(shifted, k=0.5, h=8)
    assert i0 is None and i1 is not None and i1 >= 100 and s.min() >= 0


# ---- evaluation
def test_split_is_by_unit_disjoint_deterministic():
    tr, va, te = split_units(range(1, 41), SplitConfig())
    assert not (set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te))
    assert sorted(tr + va + te) == list(range(1, 41))
    assert (tr, va, te) == split_units(range(1, 41), SplitConfig())


def test_balanced_weights_equalise_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_weights(y)
    assert math.isclose(w[y == 1].sum(), w[y == 0].sum())


def test_first_alarm_requires_persistence():
    s = np.array([0, 0.9, 0, 0.9, 0.9, 0.9])
    assert first_alarm(s, 0.5, 1) == 1 and first_alarm(s, 0.5, 3) == 5 and first_alarm(s, 2.0) is None


def test_cost_curve_and_threshold_follow_cost_ratio():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    risk = np.clip(0.35 * y + rng.normal(0.35, 0.2, 2000), 0, 1)
    cc = cost_curve(y, risk, 1, 10, np.array([0.5]))
    p = risk >= 0.5
    assert cc.fp[0] == (p & (y == 0)).sum() and cc.fn[0] == (~p & (y == 1)).sum()
    cheap_miss = choose_threshold(y, risk, ModelConfig(c_missed=1.0))
    dear_miss = choose_threshold(y, risk, ModelConfig(c_missed=50.0))
    assert dear_miss < cheap_miss
    with pytest.raises(ValueError):
        choose_threshold(y, risk, ModelConfig(threshold_strategy="nope"))


def test_alarm_table_lead_and_status():
    f = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 101), "rul": np.arange(99, -1, -1),
                      "s": np.r_[np.zeros(60), np.ones(40)]})
    row = alarm_table(f, "s", 0.5, 2, early_limit=90).iloc[0]
    assert row.alarm_cycle == 62 and row.lead == 38 and row.status == "ok"
    assert alarm_table(f, "s", 0.5, 2, early_limit=10).iloc[0].status == "premature"
    assert alarm_table(f, "s", 5.0, 1, 90).iloc[0].status == "missed"


def test_naive_rul_clipped():
    np.testing.assert_allclose(naive_rul(np.array([0, 100, 500]), 200, 120), [120, 100, 0])


# ---- pipeline / model
def test_experiment_uses_unit_level_split_and_val_threshold(exp):
    s = exp.splits
    assert not (set(s["train"]) & set(s["val"]) | set(s["train"]) & set(s["test"]) | set(s["val"]) & set(s["test"]))
    assert exp.model.threshold == pytest.approx(
        choose_threshold(*_val_xy(exp), exp.model.model_cfg))


def _val_xy(exp):
    v = exp.scored[exp.scored.unit.isin(exp.splits["val"])]
    return v["y"], v["risk"]


def test_model_beats_naive_baseline_on_heldout(exp):
    r = exp.metrics["rul"]
    assert r["model"]["mae"] < r["naive_age_baseline"]["mae"]
    c = exp.metrics["classification"]
    assert c["pr_auc"] > c["prevalence"] and 0 <= c["precision"] <= 1


def test_model_roundtrip_and_predict_new_readings(exp, fleet, tmp_path):
    p = exp.model.save(tmp_path / "m.joblib")
    m2 = MaintenanceModel.load(p)
    new = fleet[fleet.unit == 5][["unit", "cycle", *SENSORS]]
    a, b = exp.model.predict(new), m2.predict(new)
    pd.testing.assert_frame_equal(a, b)
    assert a["risk"].between(0, 1).all() and a["rul_pred"].between(0, 120).all()
    assert set(a["tier"]) <= {"normal", "watch", "plan_maintenance", "critical"}
    with pytest.raises(DataValidationError):
        m2.predict(new.drop(columns=["rpm"]))


def test_action_tier_ordering():
    c = ModelConfig()
    assert action_tier(0.01, 100, 0.5, c) == "normal"
    assert action_tier(0.3, 100, 0.5, c) == "watch"
    assert action_tier(0.6, 100, 0.5, c) == "plan_maintenance"
    assert action_tier(0.9, 100, 0.5, c) == "critical"
    assert action_tier(0.1, 5, 0.5, c) == "critical"


def test_gradient_boosting_and_random_forest_both_train(fleet):
    for mt in ("gb", "rf"):
        e = run_experiment(SIM, FeatureConfig(), ModelConfig(model_type=mt, n_estimators=20), SplitConfig(),
                           data=fleet, with_cv=False, with_permutation=False)
        assert e.metrics["classification"]["pr_auc"] > 0


# ---- report and CLI
def test_html_report_inline_charts_and_disclaimer(exp):
    te = exp.scored[exp.scored.unit.isin(exp.splits["test"])]
    html = build_report(te, exp.model, units=exp.splits["test"][:2], title="T <x>")
    assert html.count("data:image/png;base64,") == 2 and "&lt;x&gt;" in html
    assert "not evidence of real-world performance" in html and "Action tiers" in html


def test_cli_end_to_end(tmp_path, capsys, exp, fleet):
    csv = tmp_path / "new.csv"
    fleet[fleet.unit.isin([1, 2])][["unit", "cycle", *SENSORS]].to_csv(csv, index=False)
    model = exp.model.save(tmp_path / "m.joblib")
    out = tmp_path / "pred.csv"
    assert main(["predict", "--model", str(model), "--input", str(csv), "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "est_rul" in text and "synthetic" in text.lower()
    assert {"risk", "rul_pred", "tier"} <= set(pd.read_csv(out).columns)
    rep = tmp_path / "r.html"
    assert main(["report", "--model", str(model), "--input", str(csv), "--out", str(rep)]) == 0
    assert rep.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert main(["predict", "--model", str(model), "--input", str(tmp_path / "missing.csv")]) == 2
    bad = tmp_path / "bad.csv"; pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    assert main(["predict", "--model", str(model), "--input", str(bad)]) == 2


def test_cli_simulate(tmp_path, capsys):
    out = tmp_path / "s.csv"
    assert main(["simulate", "--units", "3", "--out", str(out), "--strip-labels"]) == 0
    df = pd.read_csv(out)
    assert "rul" not in df.columns and df.unit.nunique() == 3
