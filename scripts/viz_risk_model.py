"""Visualise the Guardiola-zone expected-xT risk model.

Reads output/pff/fifa-wc-2022/pass_residuals.csv and produces two figures:

  output/risk_model_guardiola_overview.png
    Single pitch heatmap — mean LBP xT_gain per zone, across all game states.
    Zone number + value + LBP count labelled inside each rectangle.

  output/risk_model_guardiola_by_gamestate.png
    Row of 5 pitch heatmaps, one per game-state bucket.
    Shared colour scale so buckets are directly comparable.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mplsoccer import Pitch  # type: ignore[import-untyped]

from turf.zone import GUARDIOLA_ZONE_BOUNDS, GUARDIOLA_ZONE_NUMBER

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESIDUALS = ROOT / "output" / "pff" / "fifa-wc-2022" / "pass_residuals.csv"
OUT_DIR = ROOT / "output"

# ── pitch constants ───────────────────────────────────────────────────────────

PITCH_TYPE = "skillcorner"
PITCH_LEN = 105.0
PITCH_WID = 68.0

GAME_STATE_ORDER = ["losing_2+", "losing_1", "level", "winning_1", "winning_2+"]
GAME_STATE_LABELS = {
    "losing_2+": "Losing 2+",
    "losing_1":  "Losing 1",
    "level":     "Level",
    "winning_1": "Winning 1",
    "winning_2+": "Winning 2+",
}

CMAP = "RdYlGn"  # diverging: red = below 0, green = above 0

# z20 (pg_gk_att) is excluded: LBPs starting from inside the attacking box
# structurally produce negative xT gain (already at high xT), making it a
# qualitatively different event that distorts the colour scale.
EXCLUDED_ZONES = {"pg_gk_att"}


def _make_pitch(ax: plt.Axes) -> Pitch:
    pitch = Pitch(
        pitch_type=PITCH_TYPE,
        pitch_length=PITCH_LEN,
        pitch_width=PITCH_WID,
        pitch_color="#1a1a2e",
        line_color="white",
        linewidth=1.2,
    )
    pitch.draw(ax=ax)
    return pitch


def _draw_zones(
    ax: plt.Axes,
    zone_values: dict[str, float],
    norm: Normalize,
    cmap: str,
    show_labels: bool = True,
    count_by_zone: dict[str, int] | None = None,
) -> None:
    """Draw coloured zone rectangles on *ax*."""
    cm = plt.get_cmap(cmap)
    for label, (x0, x1, y0, y1) in GUARDIOLA_ZONE_BOUNDS.items():
        val = zone_values.get(label, float("nan"))
        color = cm(norm(val)) if not math.isnan(val) else "#333355"  # nan → dark
        alpha = 0.75

        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, edgecolor="white", linewidth=0.8,
            alpha=alpha, zorder=2,
        ))

        if not show_labels:
            continue

        zid = GUARDIOLA_ZONE_NUMBER[label]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        is_gk = label in ("pg_gk_def", "pg_gk_att")
        fs_id = 6.5 if is_gk else 7.5

        val_str = f"{val:.4f}" if not math.isnan(val) else "n/a"
        count_str = ""
        if count_by_zone:
            n = count_by_zone.get(label, 0)
            count_str = f"\n(n={n})"

        ax.text(
            cx, cy + (0.5 if not is_gk else 1.5),
            f"z{zid}",
            ha="center", va="center", fontsize=fs_id + 0.5,
            color="white", fontweight="bold", zorder=3,
        )
        ax.text(
            cx, cy - (1.5 if not is_gk else 0.5),
            val_str + count_str,
            ha="center", va="center", fontsize=5.5,
            color="white", zorder=3,
        )


def _add_colorbar(fig: plt.Figure, ax: plt.Axes, norm: Normalize, cmap: str) -> None:
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
    cb.set_label("Mean LBP xT Gain", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")


# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(RESIDUALS)
lbp = df[df["is_line_breaking"].astype(bool)].copy()
lbp_filt = lbp[~lbp["zone_guardiola"].isin(EXCLUDED_ZONES)].copy()

# Mean xT_gain per zone across all game states (excluded zones show as NaN)
zone_mean = lbp_filt.groupby("zone_guardiola")["actual_xt_gain"].mean().to_dict()
zone_count = lbp_filt.groupby("zone_guardiola")["actual_xt_gain"].count().astype(int).to_dict()

# Mean xT_gain per zone × game-state bucket
zone_gs_mean = (
    lbp_filt.groupby(["game_state_bucket", "zone_guardiola"])["actual_xt_gain"]
    .mean()
    .unstack(level="zone_guardiola")
    .reindex(GAME_STATE_ORDER)
    .to_dict(orient="index")
)

all_vals = [v for v in zone_mean.values() if not math.isnan(v)]
# Symmetric around 0 so the diverging colormap is centred correctly.
vlim = max(abs(min(all_vals)), abs(max(all_vals)))
norm_global = Normalize(vmin=-vlim, vmax=vlim)


# ── Figure 1: overview ────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(13, 7))
fig1.patch.set_facecolor("#0d0d1a")
ax1.set_facecolor("#0d0d1a")

_make_pitch(ax1)
_draw_zones(
    ax1, zone_mean, norm_global, CMAP, show_labels=True, count_by_zone=zone_count
)
_add_colorbar(fig1, ax1, norm_global, CMAP)

ax1.set_title(
    "Expected xT Gain per Guardiola Zone — WC 2022 LBPs (all game states)",
    color="white", fontsize=12, pad=10,
)
ax1.text(
    0, -36.5,
    "Attacking direction →    |    zone number = z1–z20 (see draw_zones.py)",
    color="#aaaaaa", fontsize=7, ha="center",
)

out1 = OUT_DIR / "risk_model_guardiola_overview.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"Saved: {out1}")
plt.close(fig1)


# ── Figure 2: per game-state ──────────────────────────────────────────────────

fig2, axes2d = plt.subplots(2, 3, figsize=(24, 14))
fig2.patch.set_facecolor("#0d0d1a")

# Flatten: first 5 cells for game states, last cell for colorbar
axes_flat = axes2d.flatten()
for ax, gs in zip(axes_flat[:5], GAME_STATE_ORDER, strict=False):
    ax.set_facecolor("#0d0d1a")
    _make_pitch(ax)
    gs_vals = zone_gs_mean.get(gs, {})
    _draw_zones(ax, gs_vals, norm_global, CMAP, show_labels=True)
    ax.set_title(GAME_STATE_LABELS[gs], color="white", fontsize=13, pad=8)

# Sixth panel: blank with just the colorbar
axes_flat[5].set_visible(False)
_add_colorbar(fig2, axes_flat[4], norm_global, CMAP)

fig2.suptitle(
    "Expected xT Gain per Guardiola Zone by Game State — WC 2022 LBPs",
    color="white", fontsize=14, y=1.01,
)
fig2.tight_layout(pad=2.0)

out2 = OUT_DIR / "risk_model_guardiola_by_gamestate.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved: {out2}")
plt.close(fig2)
