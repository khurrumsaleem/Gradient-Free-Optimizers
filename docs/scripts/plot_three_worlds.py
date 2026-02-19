"""Generate a polished infographic showing the mixed search-space feature.

Discrete, continuous, and categorical dimensions coexisting in a single
search_space dictionary.

Output: mixed_search_spaces_diagram.png (300 DPI)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Color palette (VS Code dark theme inspired)
# ---------------------------------------------------------------------------
BG_COLOR = "#FFFFFF"
CODE_BG = "#282C34"
SUBTLE_BG = "#F5F5F8"

WHITE = "#2D2D2D"
GRAY = "#6B7280"
LIGHT = "#ABB2BF"

BLUE = "#61AFEF"
GREEN = "#98C379"
ORANGE = "#D19A66"
PURPLE = "#C678DD"
RED = "#E06C75"
YELLOW = "#E5C07B"
CYAN = "#56B6C2"

MONO_FONT = "DejaVu Sans Mono"
SANS_FONT = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("auto")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Code block background
# ---------------------------------------------------------------------------
CODE_LEFT = 0.15
CODE_RIGHT = 0.85
CODE_TOP = 0.96
CODE_BOTTOM = 0.65

code_bg = mpatches.FancyBboxPatch(
    (CODE_LEFT, CODE_BOTTOM),
    CODE_RIGHT - CODE_LEFT,
    CODE_TOP - CODE_BOTTOM,
    boxstyle=mpatches.BoxStyle.Round(pad=0.012, rounding_size=0.015),
    facecolor=CODE_BG,
    edgecolor="#3E4451",
    linewidth=1.0,
    transform=ax.transAxes,
    zorder=2,
)
ax.add_patch(code_bg)

# ---------------------------------------------------------------------------
# Syntax-highlighted code rendering
# ---------------------------------------------------------------------------
CODE_FONT_SIZE = 12.5
CHAR_W = 0.0093
CODE_X0 = CODE_LEFT + 0.03
CODE_Y_TOP = CODE_TOP - 0.04
CODE_LINE_H = 0.052

# The code snippet, matching the exact original formatting:
#   search_space = {
#       "x":      np.arange(-1, 7, 1),
#       "y":      (-5.0, 5.0),
#       "kernel": ["linear", "rbf", "poly"],
#   }

code_lines = [
    # line 0: search_space = {
    [
        ("search_space", RED),
        (" = {", LIGHT),
    ],
    # line 1: "x":      np.arange(-1, 7, 1),
    [
        ("    ", None),
        ('"x"', GREEN),
        (":", LIGHT),
        ("      ", None),
        ("np", CYAN),
        (".", LIGHT),
        ("arange", BLUE),
        ("(", LIGHT),
        ("-1", ORANGE),
        (", ", LIGHT),
        ("7", ORANGE),
        (", ", LIGHT),
        ("1", ORANGE),
        ("),", LIGHT),
    ],
    # line 2: "y":      (-5.0, 5.0),
    [
        ("    ", None),
        ('"y"', GREEN),
        (":", LIGHT),
        ("      ", None),
        ("(", LIGHT),
        ("-5.0", ORANGE),
        (", ", LIGHT),
        ("5.0", ORANGE),
        ("),", LIGHT),
    ],
    # line 3: "kernel": ["linear", "rbf", "poly"],
    [
        ("    ", None),
        ('"kernel"', GREEN),
        (":", LIGHT),
        (" ", None),
        ("[", LIGHT),
        ('"linear"', GREEN),
        (", ", LIGHT),
        ('"rbf"', GREEN),
        (", ", LIGHT),
        ('"poly"', GREEN),
        ("],", LIGHT),
    ],
    # line 4: }
    [
        ("}", LIGHT),
    ],
]

# Render code and track horizontal extent of the "value" part for each data line.
# The value starts after the whitespace following the colon.
# We record (value_start_x, value_end_x, line_y) for lines 1, 2, 3.
value_extents = {}

for line_idx, tokens in enumerate(code_lines):
    y = CODE_Y_TOP - line_idx * CODE_LINE_H
    x = CODE_X0

    # For data lines, detect where the value part begins.
    # The value starts at the first non-whitespace token after the colon.
    past_colon = False
    value_started = False

    for text, color in tokens:
        # Track when we pass the colon
        if color == LIGHT and text.strip() == ":":
            past_colon = True

        if past_colon and color is None and not value_started:
            # This is whitespace between colon and value -- skip silently
            pass
        elif past_colon and color is not None and not value_started:
            value_started = True
            if line_idx in (1, 2, 3):
                value_extents[line_idx] = {"x_start": x, "y": y}

        if color is None:
            x += len(text) * CHAR_W
            continue

        ax.text(
            x,
            y,
            text,
            fontsize=CODE_FONT_SIZE,
            fontfamily=MONO_FONT,
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            zorder=3,
        )
        x += len(text) * CHAR_W

        # Update end position for value tracking
        if line_idx in (1, 2, 3) and value_started:
            value_extents[line_idx]["x_end"] = x

# ---------------------------------------------------------------------------
# Visual zone: three sub-areas side by side below the code block
# ---------------------------------------------------------------------------
VIS_TOP = 0.52
VIS_BOTTOM = 0.04
VIS_HEIGHT = VIS_TOP - VIS_BOTTOM

COL_WIDTH = 0.26
COL_GAP = 0.03
total_w = 3 * COL_WIDTH + 2 * COL_GAP
COL_X0 = (1.0 - total_w) / 2

columns = []
for i in range(3):
    cx = COL_X0 + i * (COL_WIDTH + COL_GAP)
    columns.append((cx, VIS_BOTTOM, COL_WIDTH, VIS_HEIGHT))

LABEL_COLORS = [BLUE, PURPLE, GREEN]
LABEL_NAMES = ["Discrete", "Continuous", "Categorical"]

# Subtle background rectangles for each column
for i, (cx, cy, cw, ch) in enumerate(columns):
    bg = mpatches.FancyBboxPatch(
        (cx, cy),
        cw,
        ch,
        boxstyle=mpatches.BoxStyle.Round(pad=0.008, rounding_size=0.015),
        facecolor="#EDEDF0",
        edgecolor="none",
        alpha=1.0,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(bg)

    # Subtle accent line at top of column
    accent_pad = 0.03
    ax.plot(
        [cx + accent_pad, cx + cw - accent_pad],
        [cy + ch - 0.005, cy + ch - 0.005],
        color=LABEL_COLORS[i],
        lw=2.0,
        alpha=0.35,
        transform=ax.transAxes,
        zorder=2,
        solid_capstyle="round",
    )

    # Column label
    ax.text(
        cx + cw / 2,
        cy + ch - 0.035,
        LABEL_NAMES[i],
        fontsize=15,
        fontweight="bold",
        fontfamily=SANS_FONT,
        color=LABEL_COLORS[i],
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=3,
    )

# ---------------------------------------------------------------------------
# Connection arrows from each code line to its visual column
# ---------------------------------------------------------------------------
ARROW_BOTTOM_Y = VIS_TOP + 0.015
arrow_targets_x = [cx + cw / 2 for (cx, cy, cw, ch) in columns]

# Each arrow originates from the end of its value expression at the
# y-height of its code line, creating a clear visual link.
line_to_col = {1: 0, 2: 1, 3: 2}  # code line index -> column index
arc_rads = [-0.3, 0.0, 0.15]  # curvature tuned per arrow path

for line_idx, col_idx in line_to_col.items():
    ext = value_extents[line_idx]
    src_x = ext["x_end"] + 0.008
    src_y = ext["y"]

    ax.annotate(
        "",
        xy=(arrow_targets_x[col_idx], ARROW_BOTTOM_Y),
        xytext=(src_x, src_y),
        arrowprops=dict(
            arrowstyle="->,head_length=0.35,head_width=0.18",
            color=LABEL_COLORS[col_idx],
            lw=1.3,
            alpha=0.75,
            connectionstyle=f"arc3,rad={arc_rads[col_idx]}",
        ),
        transform=ax.transAxes,
        zorder=2,
    )

# ---------------------------------------------------------------------------
# DISCRETE visualization (left column): dots on a number line
# ---------------------------------------------------------------------------
disc_cx, disc_cy, disc_cw, disc_ch = columns[0]
disc_center_x = disc_cx + disc_cw / 2
disc_center_y = disc_cy + disc_ch * 0.45

nl_left = disc_cx + 0.025
nl_right = disc_cx + disc_cw - 0.025
nl_y = disc_center_y

ax.plot(
    [nl_left, nl_right],
    [nl_y, nl_y],
    color=GRAY,
    lw=1.0,
    alpha=0.4,
    transform=ax.transAxes,
    zorder=3,
)

# Dot positions: -1 to 6 (8 values from np.arange(-1, 7, 1))
values = list(range(-1, 7))
n_vals = len(values)
for j, v in enumerate(values):
    frac = j / (n_vals - 1)
    x_pos = nl_left + frac * (nl_right - nl_left)

    # Glow behind dot
    ax.plot(
        x_pos,
        nl_y,
        "o",
        color=BLUE,
        markersize=13,
        alpha=0.15,
        markeredgecolor="none",
        transform=ax.transAxes,
        zorder=3,
    )
    # Dot itself
    ax.plot(
        x_pos,
        nl_y,
        "o",
        color=BLUE,
        markersize=7.5,
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.0,
        transform=ax.transAxes,
        zorder=4,
    )

    # Label endpoints and a few middle values for legibility
    if v in (-1, 1, 3, 6):
        ax.text(
            x_pos,
            nl_y - 0.04,
            str(v),
            fontsize=8,
            fontfamily=MONO_FONT,
            color=GRAY,
            ha="center",
            va="center",
            transform=ax.transAxes,
            zorder=4,
        )

# Caption below
ax.text(
    disc_center_x,
    disc_cy + 0.055,
    "Fixed integer steps",
    fontsize=9,
    fontfamily=SANS_FONT,
    color=GRAY,
    style="italic",
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=3,
)

# ---------------------------------------------------------------------------
# CONTINUOUS visualization (center column): gradient bar
# ---------------------------------------------------------------------------
cont_cx, cont_cy, cont_cw, cont_ch = columns[1]
cont_center_x = cont_cx + cont_cw / 2
cont_center_y = cont_cy + cont_ch * 0.45

bar_left = cont_cx + 0.025
bar_right = cont_cx + cont_cw - 0.025
bar_h = 0.045
bar_bottom = cont_center_y - bar_h / 2

# Gradient via inset axes
inv = ax.transAxes + fig.transFigure.inverted()
p_bl = inv.transform((bar_left, bar_bottom))
p_tr = inv.transform((bar_right, bar_bottom + bar_h))

ax_grad = fig.add_axes(
    [p_bl[0], p_bl[1], p_tr[0] - p_bl[0], p_tr[1] - p_bl[1]],
    facecolor="none",
)
ax_grad.axis("off")

cmap = LinearSegmentedColormap.from_list(
    "cont_grad", ["#F3E5F5", PURPLE, "#E8B4F8", PURPLE, "#F3E5F5"]
)
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax_grad.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])

# Rounded border
grad_border = mpatches.FancyBboxPatch(
    (bar_left, bar_bottom),
    bar_right - bar_left,
    bar_h,
    boxstyle=mpatches.BoxStyle.Round(pad=0.0, rounding_size=0.008),
    facecolor="none",
    edgecolor=PURPLE,
    linewidth=1.0,
    alpha=0.5,
    transform=ax.transAxes,
    zorder=5,
)
ax.add_patch(grad_border)

# End labels
ax.text(
    bar_left + 0.005,
    bar_bottom - 0.03,
    "-5.0",
    fontsize=9,
    fontfamily=MONO_FONT,
    color=GRAY,
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=4,
)
ax.text(
    bar_right - 0.005,
    bar_bottom - 0.03,
    "5.0",
    fontsize=9,
    fontfamily=MONO_FONT,
    color=GRAY,
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=4,
)

# Double-headed arrow with "any value" label
mid_x = (bar_left + bar_right) / 2
ax.annotate(
    "",
    xy=(bar_right - 0.015, bar_bottom - 0.058),
    xytext=(bar_left + 0.015, bar_bottom - 0.058),
    arrowprops=dict(
        arrowstyle="<->,head_length=0.2,head_width=0.1",
        color=PURPLE,
        lw=1.0,
        alpha=0.45,
    ),
    transform=ax.transAxes,
    zorder=4,
)
ax.text(
    mid_x,
    bar_bottom - 0.058,
    "any value",
    fontsize=8,
    fontfamily=SANS_FONT,
    color=PURPLE,
    alpha=0.65,
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=5,
    bbox=dict(facecolor=SUBTLE_BG, edgecolor="none", pad=2),
)

# Caption below
ax.text(
    cont_center_x,
    cont_cy + 0.055,
    "Any float in range",
    fontsize=9,
    fontfamily=SANS_FONT,
    color=GRAY,
    style="italic",
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=3,
)

# ---------------------------------------------------------------------------
# CATEGORICAL visualization (right column): tag chips
# ---------------------------------------------------------------------------
cat_cx, cat_cy, cat_cw, cat_ch = columns[2]
cat_center_x = cat_cx + cat_cw / 2
cat_center_y = cat_cy + cat_ch * 0.45

categories = ["linear", "rbf", "poly"]
chip_fill = "#E8F5E9"
chip_edge = GREEN
chip_w = 0.072
chip_h = 0.05
chip_gap = 0.012
total_chip_w = len(categories) * chip_w + (len(categories) - 1) * chip_gap
chip_x0 = cat_center_x - total_chip_w / 2

for j, cat_name in enumerate(categories):
    cx_chip = chip_x0 + j * (chip_w + chip_gap)
    cy_chip = cat_center_y - chip_h / 2

    chip = mpatches.FancyBboxPatch(
        (cx_chip, cy_chip),
        chip_w,
        chip_h,
        boxstyle=mpatches.BoxStyle.Round(pad=0.005, rounding_size=0.01),
        facecolor=chip_fill,
        edgecolor=chip_edge,
        linewidth=1.3,
        alpha=0.85,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(chip)

    ax.text(
        cx_chip + chip_w / 2,
        cat_center_y,
        cat_name,
        fontsize=10.5,
        fontfamily=MONO_FONT,
        fontweight="bold",
        color=GREEN,
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=5,
    )

# Caption below
ax.text(
    cat_center_x,
    cat_cy + 0.055,
    "Unordered choices",
    fontsize=9,
    fontfamily=SANS_FONT,
    color=GRAY,
    style="italic",
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=3,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (
    "/home/me/github-workspace/002-gfo-stack/"
    "Gradient-Free-Optimizers/mixed_search_spaces_diagram.png"
)
fig.savefig(out_path, dpi=500, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {out_path}")
