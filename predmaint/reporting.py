"""Matplotlib charts (Agg backend) and the per-machine HTML maintenance report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import HEALTH_SENSORS, SENSORS  # noqa: E402
from .evaluation import first_alarm  # noqa: E402
from .model import TIER_ACTIONS, TIERS, MaintenanceModel  # noqa: E402

TIER_COLORS = {"normal": "#2a9d8f", "watch": "#e9c46a", "plan_maintenance": "#f4a261", "critical": "#d62828"}
BLUE, GREY, RED = "#264653", "#8d99ae", "#d62828"
DISCLAIMER = (
    "Synthetic demonstration data. Scores on simulated data are optimistic and are not evidence of "
    "real-world performance. Do not use this output to make real maintenance decisions."
)


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_fig(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ charts
def plot_sensors(unit_df: pd.DataFrame):
    """Raw sensors of one simulated unit (regime changes, noise, dropout gaps, degradation)."""
    fig, axes = plt.subplots(len(SENSORS), 1, figsize=(9, 8), sharex=True)
    for ax, s in zip(axes, SENSORS):
        ax.plot(unit_df["cycle"], unit_df[s], lw=0.9, color=BLUE)
        miss = unit_df["cycle"][unit_df[s].isna()]
        for c in miss:
            ax.axvline(c, color=RED, alpha=0.3, lw=0.8)
        ax.set_ylabel(s, fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Simulated unit {int(unit_df['unit'].iloc[0])}: raw sensors (red lines = dropout)", fontsize=10, loc="left")
    axes[-1].set_xlabel("cycle")
    return fig


def plot_rul(scored: pd.DataFrame, units: list[int]):
    fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, u in zip(axes, units):
        g = scored[scored["unit"] == u].sort_values("cycle")
        ax.plot(g["cycle"], g["rul_cap"], color=GREY, label="true RUL (capped)")
        ax.plot(g["cycle"], g["rul_pred"], color=BLUE, label="predicted")
        _style(ax, f"held-out unit {u} ({g['fault_mode'].iloc[0]})", "cycle", "RUL (cycles)" if ax is axes[0] else "")
    axes[0].legend(fontsize=7)
    return fig


def plot_cost(curve: pd.DataFrame, chosen: float):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(curve["threshold"], curve["cost"], color=BLUE)
    ax.axvline(chosen, color=RED, ls="--", label=f"chosen on validation units: {chosen:.2f}")
    _style(ax, "Alarm cost vs. threshold (held-out units)", "risk threshold", "cost (illustrative units)")
    ax.legend(fontsize=8)
    return fig


def plot_pr(y, risk, age_score):
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    p, r, _ = precision_recall_curve(y, risk)
    ax.plot(r, p, color=BLUE, label="gradient boosting")
    p, r, _ = precision_recall_curve(y, age_score)
    ax.plot(r, p, color=GREY, label="age-only baseline")
    _style(ax, "Precision-recall (held-out units)", "recall", "precision")
    ax.legend(fontsize=8)
    return fig


def plot_lead(tables: dict[str, pd.DataFrame], early_limit: int):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(tables)
    data = [tables[n].loc[tables[n]["status"] != "missed", "lead"].to_numpy() for n in names]
    ax.boxplot(data, tick_labels=names, vert=True)
    ax.axhline(early_limit, color=RED, ls=":", label=f"> {early_limit} cycles = premature")
    _style(ax, "Cycles before failure of first alarm (held-out units)", "", "lead time (cycles)")
    ax.legend(fontsize=8)
    return fig


def plot_importance(imp: pd.DataFrame, title: str, top: int = 12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.barh(d["feature"], d["importance"], color=BLUE)
    _style(ax, title, "importance")
    return fig


def plot_machine(g: pd.DataFrame, threshold: float, cusum_h: float, alarm_cycle: int | None):
    """Health-index trend plus alarm timeline for one machine (g sorted by cycle)."""
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2, 0.8]})
    a1.plot(g["cycle"], g["health_index"], color=BLUE, label="health index (from est. RUL)")
    if "rul" in g and g["rul"].notna().any():
        a1.plot(g["cycle"], 100 * np.clip(g["rul"], 0, None) / max(g["rul"].max(), 1) , color=GREY, lw=0.8, ls=":")
    _style(a1, "Health index (100 = healthy, 0 = predicted failure)", "", "index")
    a1.set_ylim(-3, 103)
    a2.plot(g["cycle"], g["risk"], color=RED, label="failure risk")
    a2.axhline(threshold, color="k", ls="--", lw=0.8, label=f"alarm threshold {threshold:.2f}")
    _style(a2, "Failure-within-horizon risk", "", "risk")
    a2.legend(fontsize=7, loc="upper left")
    for ax in (a1, a2):
        if alarm_cycle is not None:
            ax.axvline(alarm_cycle, color="k", lw=1.0, alpha=0.6)
    for t in TIERS:
        m = g["tier"] == t
        a3.scatter(g["cycle"][m], np.zeros(int(m.sum())), marker="|", s=250, color=TIER_COLORS[t], label=t)
    a3.set_yticks([])
    a3.set_xlabel("cycle")
    a3.set_title("Alarm timeline (action tier per cycle)", fontsize=10, loc="left")
    a3.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.9))
    return fig


# ------------------------------------------------------------------ HTML report
_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.45}
h1{font-size:1.5rem}h2{border-bottom:1px solid #ccd;padding-bottom:.2rem;margin-top:2.2rem}
.warn{background:#fff3cd;border:1px solid #e0c36a;padding:.6rem .8rem;border-radius:6px;font-size:.9rem}
table{border-collapse:collapse;margin:.6rem 0}td,th{border:1px solid #ccd;padding:.3rem .6rem;text-align:left;font-size:.9rem}
.tier{display:inline-block;padding:.1rem .5rem;border-radius:4px;color:#fff;font-weight:600}
img{max-width:100%}
"""


def _tier_badge(t: str) -> str:
    return f'<span class="tier" style="background:{TIER_COLORS[t]}">{html.escape(t)}</span>'


def machine_summary(g: pd.DataFrame, model: MaintenanceModel) -> dict:
    g = g.sort_values("cycle")
    i = first_alarm(g["risk"].to_numpy(), model.threshold, model.model_cfg.persistence)
    last = g.iloc[-1]
    return {
        "unit": int(last["unit"]),
        "n_cycles": int(len(g)),
        "last_cycle": int(last["cycle"]),
        "risk": float(last["risk"]),
        "rul_pred": float(last["rul_pred"]),
        "tier": str(last["tier"]),
        "health_index": float(last["health_index"]),
        "first_alarm_cycle": None if i is None else int(g["cycle"].iloc[i]),
    }


def build_report(scored: pd.DataFrame, model: MaintenanceModel, units: list[int] | None = None,
                 title: str = "Predictive maintenance report") -> str:
    """Self-contained HTML (inline CSS, base64 PNG charts), one section per machine."""
    units = sorted(scored["unit"].unique()) if units is None else units
    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style></head><body><h1>{html.escape(title)}</h1>",
        f"<p class='warn'><b>Read first.</b> {html.escape(DISCLAIMER)}</p>",
        "<h2>Fleet overview</h2><table><tr><th>Unit</th><th>Last cycle</th><th>Tier</th><th>Risk</th>"
        "<th>Est. RUL (cycles, capped)</th><th>First alarm cycle</th></tr>",
    ]
    sums = [machine_summary(scored[scored["unit"] == u], model) for u in units]
    for s in sums:
        fa = "-" if s["first_alarm_cycle"] is None else s["first_alarm_cycle"]
        parts.append(
            f"<tr><td><a href='#u{s['unit']}'>{s['unit']}</a></td><td>{s['last_cycle']}</td><td>{_tier_badge(s['tier'])}</td>"
            f"<td>{s['risk']:.2f}</td><td>{s['rul_pred']:.0f}</td><td>{fa}</td></tr>"
        )
    parts.append("</table>")
    for s in sums:
        g = scored[scored["unit"] == s["unit"]].sort_values("cycle")
        img = fig_to_b64(plot_machine(g, model.threshold, model.cusum_h, s["first_alarm_cycle"]))
        parts.append(
            f"<h2 id='u{s['unit']}'>Machine {s['unit']}</h2>"
            f"<p>Latest tier: {_tier_badge(s['tier'])} (risk {s['risk']:.2f}, est. RUL {s['rul_pred']:.0f} cycles, "
            f"health index {s['health_index']:.0f}). Recommended action: {html.escape(TIER_ACTIONS[s['tier']])}</p>"
            f"<img alt='health index, risk and alarm timeline for machine {s['unit']}' src='data:image/png;base64,{img}'>"
        )
    parts.append("<h2>Action tiers</h2><table><tr><th>Tier</th><th>Rule</th><th>Suggested action</th></tr>")
    c = model.model_cfg
    rules = {
        "normal": f"risk &lt; {c.watch_ratio * model.threshold:.2f}",
        "watch": f"risk &ge; {c.watch_ratio * model.threshold:.2f}",
        "plan_maintenance": f"risk &ge; alarm threshold {model.threshold:.2f}",
        "critical": f"risk &ge; {c.critical_risk} or est. RUL &le; {c.critical_rul:.0f}",
    }
    for t in TIERS:
        parts.append(f"<tr><td>{_tier_badge(t)}</td><td>{rules[t]}</td><td>{html.escape(TIER_ACTIONS[t])}</td></tr>")
    parts.append("</table><p>RUL estimates are capped at "
                 f"{model.feature_cfg.rul_cap} cycles. Tiers are illustrative rules, not validated maintenance policy.</p></body></html>")
    return "".join(parts)


def write_report(scored, model, path: str | Path, units=None, title="Predictive maintenance report") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(scored, model, units, title), encoding="utf-8")
    return path
