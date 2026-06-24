"""Quick zone scheme visualizer — for confirmation only, not part of turf package.

Coordinate system: skillcorner — origin at pitch centre.
  x: -52.5 to +52.5  (goal-to-goal, left goal = negative)
  y: -34.0 to +34.0  (touchline-to-touchline, bottom = negative)
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mplsoccer import Pitch

PITCH_TYPE = "skillcorner"
PITCH_LEN = 105.0
PITCH_WID = 68.0

HL = PITCH_LEN / 2       # 52.5  (half-length)
HW = PITCH_WID / 2       # 34.0  (half-width)

PEN_DEPTH = 16.5          # penalty area depth from goal line
PEN_HALF_W = 40.32 / 2   # 20.16 — half-width of penalty area (= y boundary in skillcorner)
KC_R = 9.15               # kickoff-circle radius → also the half-space y boundary in skillcorner

# Inner-column width for Van Gaal (and Guardiola flank middle zones):
# (pitch_length − 2 × penalty_depth) / 4  =  72 / 4  =  18 m
INNER_COL = (PITCH_LEN - 2 * PEN_DEPTH) / 4   # 18.0


def make_pitch(ax: plt.Axes) -> Pitch:
    pitch = Pitch(
        pitch_type=PITCH_TYPE,
        pitch_length=PITCH_LEN,
        pitch_width=PITCH_WID,
        pitch_color="#3a7d44",
        line_color="white",
        linewidth=1.5,
    )
    pitch.draw(ax=ax)
    return pitch


fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.patch.set_facecolor("#1a1a2e")

# ── 1. Simple Thirds (3 zones) ────────────────────────────────────────────────
ax = axes[0]
ax.set_title("Simple Thirds  (3 zones)", color="white", fontsize=13, pad=10)
make_pitch(ax)

bounds_3 = [-HL, -HL / 3, HL / 3, HL]
clr_3 = ["#e74c3c", "#f39c12", "#27ae60"]
lbl_3 = ["Defensive\nThird", "Middle\nThird", "Attacking\nThird"]
for i in range(3):
    ax.add_patch(patches.Rectangle(
        (bounds_3[i], -HW), bounds_3[i + 1] - bounds_3[i], PITCH_WID,
        alpha=0.45, facecolor=clr_3[i], edgecolor="white", lw=1.5, zorder=2,
    ))
    ax.text(
        (bounds_3[i] + bounds_3[i + 1]) / 2, 0, lbl_3[i],
        ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3,
    )

# ── 2. Van Gaal (6×3 = 18 zones) ─────────────────────────────────────────────
# Columns (x): outer 2 align with penalty depth; inner 4 split equally at 18 m.
# Rows    (y): outer 2 align with penalty area width; centre is wide.
ax = axes[1]
ax.set_title("Van Gaal  (6×3 = 18 zones)", color="white", fontsize=13, pad=10)
make_pitch(ax)

col_vg = [-HL, -HL + PEN_DEPTH, -INNER_COL, 0, INNER_COL, HL - PEN_DEPTH, HL]
# =    [-52.5,    -36.0,          -18.0,      0,  18.0,      36.0,           52.5]

row_vg = [-HW, -PEN_HALF_W, PEN_HALF_W, HW]
# =    [-34,   -20.16,       20.16,      34]

clr_vg = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
n = 1
for c in range(6):
    for r in range(3):
        x0, x1 = col_vg[c], col_vg[c + 1]
        y0, y1 = row_vg[r], row_vg[r + 1]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            alpha=0.4, facecolor=clr_vg[c], edgecolor="white", lw=1, zorder=2,
        ))
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, str(n),
            ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=3,
        )
        n += 1

# ── 3. Guardiola (20 zones) ───────────────────────────────────────────────────
# Vertical lanes (y): PEN_HALF_W and KC_R as boundaries.
#   L-Flank   : y ∈ [-34,    -20.16]
#   L-HalfSpc : y ∈ [-20.16, -9.15 ]
#   Center    : y ∈ [-9.15,  +9.15 ]
#   R-HalfSpc : y ∈ [+9.15,  +20.16]
#   R-Flank   : y ∈ [+20.16, +34   ]
#
# Row boundaries (x):
#   Flanks (6 zones): same x-boundaries as Van Gaal columns
#     [-52.5, -36, -18, 0, 18, 36, 52.5]
#   Inner lanes (2 middle zones): only penalty depth and halfway
#     x ∈ [-36, 0] and x ∈ [0, +36]
#   GK boxes (1 zone each): x ∈ [-52.5,-36] and x ∈ [36, 52.5],
#     y spanning L-HalfSpc → R-HalfSpc (i.e. -20.16 to +20.16)

ax = axes[2]
ax.set_title("Guardiola  (20 zones)", color="white", fontsize=13, pad=10)
make_pitch(ax)

lane_y = [-HW, -PEN_HALF_W, -KC_R, KC_R, PEN_HALF_W, HW]
# =     [-34,   -20.16,     -9.15, 9.15,  20.16,      34]

lane_names = ["L-Flank", "L-Half\nSpace", "Center\nChannel", "R-Half\nSpace", "R-Flank"]
lane_colors = ["#e67e22", "#3498db", "#27ae60", "#3498db", "#e67e22"]

# Flank rows: same x-cuts as Van Gaal (6 zones per flank)
rows_flank = col_vg   # [-52.5, -36, -18, 0, 18, 36, 52.5]

# Inner rows: middle 2 zones only (GK boxes cover the ends)
rows_inner = [-(HL - PEN_DEPTH), 0, HL - PEN_DEPTH]   # [-36, 0, 36]

n = 1

# Flanks: lanes 0 and 4, 6 rows each  →  12 zones
for lane in [0, 4]:
    y0, y1 = lane_y[lane], lane_y[lane + 1]
    for r in range(6):
        x0, x1 = rows_flank[r], rows_flank[r + 1]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            alpha=0.4, facecolor=lane_colors[lane], edgecolor="white", lw=1.5, zorder=2,
        ))
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, str(n),
            ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3,
        )
        n += 1

# Inner lanes: lanes 1, 2, 3 — 2 rows each  →  6 zones
for lane in [1, 2, 3]:
    y0, y1 = lane_y[lane], lane_y[lane + 1]
    for r in range(2):
        x0, x1 = rows_inner[r], rows_inner[r + 1]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            alpha=0.4, facecolor=lane_colors[lane], edgecolor="white", lw=1.5, zorder=2,
        ))
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, str(n),
            ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=3,
        )
        n += 1

# GK box zones: span all inner lanes at each end  →  2 zones
gk_y0, gk_y1 = lane_y[1], lane_y[4]   # -PEN_HALF_W to +PEN_HALF_W
for x0, x1, lbl in [
    (-HL, -(HL - PEN_DEPTH), "GK\nDef"),
    (HL - PEN_DEPTH, HL, "GK\nAtt"),
]:
    ax.add_patch(patches.Rectangle(
        (x0, gk_y0), x1 - x0, gk_y1 - gk_y0,
        alpha=0.55, facecolor="#9b59b6", edgecolor="white", lw=1.5, zorder=2,
    ))
    ax.text(
        (x0 + x1) / 2, 0, f"{n}\n{lbl}",
        ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3,
    )
    n += 1

total_g = n - 1

# Lane labels (outside left edge)
for lane in range(5):
    ax.text(
        -HL - 1, (lane_y[lane] + lane_y[lane + 1]) / 2,
        lane_names[lane], ha="right", va="center", fontsize=6, color="white",
    )

ax.set_title(f"Guardiola  ({total_g} zones)", color="white", fontsize=13, pad=10)

plt.tight_layout(pad=1.5)
plt.savefig("output/zone_schemes.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: output/zone_schemes.png  (Guardiola: {total_g} zones)")
