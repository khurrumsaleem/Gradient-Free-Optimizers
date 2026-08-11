"""3D scatter plot of RandomRestartHillClimbing exploring a mixed search space.

The categorical dimension (algorithm) is mapped to the Z-axis, creating
three distinct layers in 3D space. Points are colored by score.

Output: search_landscape_plot.png (300 DPI)
"""

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from gradient_free_optimizers import (
    EvolutionStrategyOptimizer,
)

# ---------------------------------------------------------------------------
# 1. Objective and search space
# ---------------------------------------------------------------------------


def objective(params):
    x = params["x"]
    y = params["y"]
    algo = params["algorithm"]

    # Each category has a different optimal region in (x, y) space
    if algo == "adam":
        score = -((x - 2) ** 2 * 0.04 + (y - 3.0) ** 2 * 0.2)
    elif algo == "sgd":
        score = -((x + 6) ** 2 * 0.04 + (y + 2.0) ** 2 * 0.2)
    else:  # rmsprop
        score = -((x - 6) ** 2 * 0.04 + (y - 0.0) ** 2 * 0.2)
    score += 0.2 * np.sin(x * 0.4) * np.cos(y * 0.6)
    return score


search_space = {
    "x": np.arange(-10, 12, 4),
    "y": (-5.0, 5.0),
    "algorithm": ["adam", "sgd", "rmsprop"],
}


# ---------------------------------------------------------------------------
# 2. Run the optimizer
# ---------------------------------------------------------------------------

opt = EvolutionStrategyOptimizer(search_space, random_state=3)
opt.search(objective, n_iter=500, verbosity=False)

df = opt.search_data

print(f"Best params : {opt.best_para}")
print(f"Best score  : {opt.best_score:.4f}")
print(f"Total evals : {len(df)}")


# ---------------------------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------------------------

scores = df["score"].values
x_vals = df["x"].values.astype(float)
y_vals = df["y"].values.astype(float)
algos = df["algorithm"].values

category_map = {"adam": 0, "sgd": 1, "rmsprop": 2}
z_vals = np.array([category_map[a] for a in algos])

# Small jitter on all axes to visually separate overlapping points.
# The optimizer converges quickly, so many evaluations share the same
# discrete x and categorical z values.
rng = np.random.default_rng(42)
n = len(z_vals)
x_jitter = x_vals + rng.uniform(-0.25, 0.25, size=n)
y_jitter = y_vals + rng.uniform(-0.08, 0.08, size=n)
z_jitter = z_vals + rng.uniform(-0.06, 0.06, size=n)

# Sort by score so the brightest (best) dots render on top
sort_order = np.argsort(scores)
x_jitter = x_jitter[sort_order]
y_jitter = y_jitter[sort_order]
z_jitter = z_jitter[sort_order]
scores_sorted = scores[sort_order]

best_idx = scores.argmax()
best_x = x_vals[best_idx]
best_y = y_vals[best_idx]
best_z = z_vals[best_idx]
best_score = scores[best_idx]

norm = Normalize(vmin=scores.min(), vmax=scores.max())


# ---------------------------------------------------------------------------
# 4. Theme constants
# ---------------------------------------------------------------------------

BG = "#FFFFFF"
AXES_BG = "#FFFFFF"
TEXT = "#2D2D2D"
GRID_COLOR = "#CCCCCC"
EDGE_COLOR = "#999999"

# RGBA for pane fill (very light, semi-transparent)
PANE_RGBA = (0.95, 0.95, 0.97, 0.4)


# ---------------------------------------------------------------------------
# 5. Build figure
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes3d.xaxis.panecolor": (0.95, 0.95, 0.97, 0.4),
        "axes3d.yaxis.panecolor": (0.95, 0.95, 0.97, 0.4),
        "axes3d.zaxis.panecolor": (0.95, 0.95, 0.97, 0.4),
    }
)

fig = plt.figure(figsize=(12, 8), facecolor=BG)
ax = fig.add_subplot(111, projection="3d", facecolor=AXES_BG)

# Scatter all sample points (small dots, no depthshade on dark BG).
# Uses sorted+jittered arrays so bright points draw on top.
scatter = ax.scatter(
    y_jitter,
    x_jitter,
    z_jitter,
    c=scores_sorted,
    cmap="jet",
    s=10,
    alpha=0.8,
    norm=norm,
    edgecolors="none",
    depthshade=False,
)

# Faint horizontal planes at each categorical level for visual separation
plane_x_range = np.array([y_vals.min() - 0.5, y_vals.max() + 0.5])
plane_y_range = np.array([x_vals.min() - 0.5, x_vals.max() + 0.5])
plane_X, plane_Y = np.meshgrid(plane_x_range, plane_y_range)
for z_level in [0, 1, 2]:
    plane_Z = np.full_like(plane_X, z_level)
    ax.plot_surface(
        plane_X,
        plane_Y,
        plane_Z,
        alpha=0.05,
        color="#CCCCCC",
        edgecolor="none",
    )


# ---------------------------------------------------------------------------
# 6. Axis labels and ticks
# ---------------------------------------------------------------------------

ax.set_xlabel("y  (continuous)", color=TEXT, fontsize=11, labelpad=10)
ax.set_ylabel("x  (discrete)", color=TEXT, fontsize=11, labelpad=10)
ax.set_zlabel("algorithm  (categorical)", color=TEXT, fontsize=11, labelpad=10)

ax.set_zticks([0, 1, 2])
ax.set_zticklabels(["adam", "sgd", "rmsprop"])


# ---------------------------------------------------------------------------
# 7. 3D pane, grid, and edge styling
# ---------------------------------------------------------------------------

# Pane fill colors
ax.xaxis.set_pane_color(PANE_RGBA)
ax.yaxis.set_pane_color(PANE_RGBA)
ax.zaxis.set_pane_color(PANE_RGBA)

# Grid and axis-line styling via _axinfo (the only reliable way for 3D)
grid_rgba = mcolors.to_rgba(GRID_COLOR, alpha=0.25)
edge_rgba = mcolors.to_rgba(EDGE_COLOR, alpha=0.6)

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.label.set_color(TEXT)
    axis._axinfo["tick"]["color"] = TEXT
    axis._axinfo["grid"]["color"] = grid_rgba
    axis._axinfo["grid"]["linewidth"] = 0.4
    axis._axinfo["axisline"]["color"] = edge_rgba
    axis._axinfo["tick"]["inward_factor"] = 0
    axis._axinfo["tick"]["outward_factor"] = 0.2

ax.tick_params(axis="x", colors=TEXT, labelsize=9)
ax.tick_params(axis="y", colors=TEXT, labelsize=9)
ax.tick_params(axis="z", colors=TEXT, labelsize=9, pad=6)

ax.view_init(elev=22, azim=-52)


# ---------------------------------------------------------------------------
# 8. Best point marker (drawn last to sit on top of the cluster)
# ---------------------------------------------------------------------------

# Using ax.plot instead of ax.scatter to sidestep 3D depth-sort issues
ax.plot(
    [best_y],
    [best_x],
    [best_z],
    marker="*",
    markersize=16,
    markerfacecolor="#FF4444",
    markeredgecolor="#333333",
    markeredgewidth=1.0,
    linestyle="none",
    zorder=100,
)


# ---------------------------------------------------------------------------
# 9. Colorbar
# ---------------------------------------------------------------------------

cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.1, aspect=18)
cbar.set_label("score", color=TEXT, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
for label in cbar.ax.get_yticklabels():
    label.set_color(TEXT)
cbar.outline.set_edgecolor(EDGE_COLOR)
cbar.outline.set_linewidth(0.5)


# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------

fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.97)

output_path = _REPO_ROOT / "search_landscape_plot.png"
fig.savefig(
    output_path,
    dpi=300,
    facecolor=BG,
    edgecolor="none",
    bbox_inches="tight",
)
plt.close(fig)

print(f"\nPlot saved to: {output_path}")
