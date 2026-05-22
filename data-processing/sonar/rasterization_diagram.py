#!/usr/bin/env python3
"""
Rasterization geometry diagram — Sonoptix sonar.

Hybrid academic style:
- closed axis box around the plot
- R_max annotation outside the sonar image
- solid slant-range arcs
- professional, moderate-weight annotation text
- extra inner space so annotations do not collide with the frame
- vector exports for sharp text/lines
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'          : 'serif',
    'font.serif'           : ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size'            : 10,
    'font.weight'          : 'normal',

    'axes.linewidth'       : 0.8,
    'axes.labelsize'       : 10,
    'axes.labelweight'     : 'normal',
    'axes.titleweight'     : 'normal',

    'xtick.labelsize'      : 8.5,
    'ytick.labelsize'      : 8.5,
    'xtick.direction'      : 'in',
    'ytick.direction'      : 'in',
    'xtick.major.size'     : 3.5,
    'ytick.major.size'     : 3.5,
    'xtick.major.width'    : 0.8,
    'ytick.major.width'    : 0.8,
    'xtick.minor.visible'  : False,
    'ytick.minor.visible'  : False,

    'legend.fontsize'      : 8.5,
    'legend.framealpha'    : 0.92,
    'legend.edgecolor'     : '0.7',

    'text.usetex'          : False,
    'mathtext.fontset'     : 'stix',
})


# ── Parameters ────────────────────────────────────────────────────────────────
W, H         = 1200, 700
DPI          = 150

R_max        = 20.0
FOV_half_deg = 45.0

x_span  = R_max * np.sin(np.radians(FOV_half_deg))
delta_x = 2 * x_span / W
delta_y = R_max / H

# Pixel-image ±45°:
# Equal image-pixel steps correspond to unequal metric steps.
d_px    = np.sqrt(delta_x**2 + delta_y**2)
x_pix45 = R_max * delta_x / d_px
y_pix45 = R_max * delta_y / d_px


# ── Paths ─────────────────────────────────────────────────────────────────────
IMG = (
    "/cluster/home/henrban/aquaculture-perception/data-processing/sonar/"
    "MOT/2024-08-20_17-14-36/frames/1724166918546318400.jpg"
)

OUT = (
    "/cluster/home/henrban/aquaculture-perception/data-processing/sonar/"
    "rasterization_diagram.png"
)


# ── Load image ────────────────────────────────────────────────────────────────
img = np.array(Image.open(IMG))


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor('white')

ax.imshow(
    img,
    origin='upper',
    extent=[-x_span, x_span, 0, R_max],
    aspect='auto',
    zorder=0,
)

# More room inside the boxed frame so annotations do not collide with edges.
ax.set_xlim(-x_span - 3.5, x_span + 3.5)
ax.set_ylim(-2.2, R_max + 4.8)


# ── Small helper for readable labels ──────────────────────────────────────────
def label_box(alpha=0.86):
    return dict(
        boxstyle='round,pad=0.18',
        fc='white',
        ec='none',
        alpha=alpha,
    )


# ── Slant-range arcs — solid, muted academic colors ───────────────────────────
RANGE_COLORS = {
    5:  '#4878CF',
    10: '#009E73',
    20: '#B47CC7',
}

theta = np.linspace(
    -np.radians(FOV_half_deg),
    np.radians(FOV_half_deg),
    350,
)

for r, col in RANGE_COLORS.items():
    ax.plot(
        r * np.sin(theta),
        r * np.cos(theta),
        color=col,
        lw=1.65,
        ls='-',
        zorder=2,
        solid_capstyle='round',
        solid_joinstyle='round',
        antialiased=True,
    )

    # Direct arc labels on the right side.
    ang_lbl = np.radians(FOV_half_deg - 8)
    ax.text(
        r * np.sin(ang_lbl) + 0.15,
        r * np.cos(ang_lbl),
        rf'$r = {r}\ \mathrm{{m}}$',
        fontsize=8.0,
        fontweight='normal',
        color=col,
        ha='left',
        va='center',
        bbox=label_box(0.80),
        zorder=5,
    )


# ── FOV boundary lines ────────────────────────────────────────────────────────
# Actual metric ±45° FOV.
for sign in (-1, 1):
    ax.plot(
        [0, sign * x_span],
        [0, x_span],
        color='0.92',
        lw=1.35,
        ls=(0, (5, 3)),
        zorder=3,
        antialiased=True,
    )

# Pixel-image apparent ±45°.
PIX_COLOR = '#CC6600'

for sign in (-1, 1):
    ax.plot(
        [0, sign * x_pix45],
        [0, y_pix45],
        color=PIX_COLOR,
        lw=1.35,
        ls=(0, (2, 2)),
        zorder=3,
        antialiased=True,
    )

# ── Direct labels for FOV line families ───────────────────────────────────────
# Label one representative line from each family instead of using a legend.

# Actual metric ±45° label, placed near the left FOV boundary.
metric_label_frac = 0.70

ax.text(
    -metric_label_frac * x_span - 0.55,
     metric_label_frac * x_span + 0.55,
    r'$\pm45^\circ$ FOV' + '\n' + r'metric',
    ha='right',
    va='center',
    fontsize=8.4,
    fontweight='normal',
    color='0.18',
    linespacing=1.25,
    bbox=label_box(0.86),
    zorder=7,
)

# Label for apparent pixel-image ±45° line family
pixel_label_frac = 0.42

ax.text(
    -pixel_label_frac * x_pix45 - 0.55,
     pixel_label_frac * y_pix45 - 0.35,
    r'$\pm45^\circ$'  + r'pixel image',
    ha='right',
    va='center',
    fontsize=8.4,
    fontweight='normal',
    color=PIX_COLOR,
    linespacing=1.25,
    bbox=label_box(0.86),
    zorder=7,
)

# ── Sonar origin ──────────────────────────────────────────────────────────────
ax.plot(
    0,
    0,
    'o',
    color='white',
    ms=5.2,
    markeredgecolor='0.15',
    markeredgewidth=0.85,
    zorder=6,
)

ax.text(
    0.35,
    -0.5,
    'Sonar origin',
    ha='left',
    va='bottom',
    fontsize=7.5,
    fontweight='normal',
    color='0.1',
    bbox=label_box(0.86),
    zorder=7,
)


# ── R_max annotation — outside image, right margin ────────────────────────────
x_rmax = x_span + 0.8

ax.annotate(
    '',
    xy=(x_rmax, R_max),
    xytext=(x_rmax, 0),
    arrowprops=dict(
        arrowstyle='<->',
        color='0.25',
        lw=1.15,
        mutation_scale=10,
    ),
    zorder=5,
)

# Small end ticks.
ax.plot(
    [x_rmax - 0.18, x_rmax + 0.18],
    [0, 0],
    color='0.25',
    lw=0.9,
    zorder=5,
)

ax.plot(
    [x_rmax - 0.18, x_rmax + 0.18],
    [R_max, R_max],
    color='0.25',
    lw=0.9,
    zorder=5,
)

ax.text(
    x_rmax + 0.35,
    R_max / 2,
    r'$R_{\max}$' + '\n' + r'$= 20\ \mathrm{m}$',
    ha='left',
    va='center',
    fontsize=9.8,
    fontweight='normal',
    color='0.15',
    linespacing=1.35,
    zorder=6,
)


# ── x_span annotations ────────────────────────────────────────────────────────
y_xspan = R_max + 0.8

xspan_arrow_kw = dict(
    arrowstyle='<->',
    color='0.30',
    lw=1.15,
    mutation_scale=10,
)

ax.annotate(
    '',
    xy=(x_span, y_xspan),
    xytext=(0, y_xspan),
    arrowprops=xspan_arrow_kw,
    zorder=4,
)

ax.annotate(
    '',
    xy=(-x_span, y_xspan),
    xytext=(0, y_xspan),
    arrowprops=xspan_arrow_kw,
    zorder=4,
)

ax.text(
    x_span / 2,
    y_xspan + 0.32,
    r'$x_{\mathrm{span}} = R_{\max}\sin(45^\circ) \approx 14.14\ \mathrm{m}$',
    ha='center',
    va='bottom',
    fontsize=10.5,
    fontweight='normal',
    color='0.15',
    zorder=5,
)

ax.text(
    -x_span / 2,
    y_xspan + 0.32,
    r'$x_{\mathrm{span}}$',
    ha='center',
    va='bottom',
    fontsize=10.5,
    fontweight='normal',
    color='0.15',
    zorder=5,
)


# ── Total width annotation ────────────────────────────────────────────────────
y_width = R_max + 2.70

ax.annotate(
    '',
    xy=(-x_span, y_width),
    xytext=(x_span, y_width),
    arrowprops=dict(
        arrowstyle='<->',
        color='0.42',
        lw=1.15,
        mutation_scale=10,
    ),
    zorder=4,
)

ax.text(
    0,
    y_width + 0.32,
    r'$2\,x_{\mathrm{span}} \approx 28.28\ \mathrm{m}$',
    ha='center',
    va='bottom',
    fontsize=10.5,
    fontweight='normal',
    color='0.22',
    zorder=5,
)


# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')

ax.set_title(
    'Processed sonar image geometry',
    fontsize=11,
    fontweight='normal',
    pad=8,
)

ax.set_xticks(np.arange(-14, 15, 2))
ax.set_yticks(np.arange(0, 21, 5))

ax.grid(
    True,
    linestyle=':',
    linewidth=0.45,
    color='0.5',
    alpha=0.42,
)

ax.axhline(0, color='0.60', lw=0.55)
ax.axvline(0, color='0.60', lw=0.55)

# Closed box around the axes.
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('0.25')


# # ── Legend ────────────────────────────────────────────────────────────────────
# legend_handles = [
#     Line2D(
#         [0],
#         [0],
#         color='0.75',
#         lw=1.35,
#         ls=(0, (5, 3)),
#         label=r'$\pm45^\circ$ FOV — metric (actual)',
#     ),
#     Line2D(
#         [0],
#         [0],
#         color=PIX_COLOR,
#         lw=1.35,
#         ls=(0, (2, 2)),
#         label=r'$\pm45^\circ$ — pixel image (apparent)',
#     ),
# ] + [
#     Line2D(
#         [0],
#         [0],
#         color=col,
#         lw=1.65,
#         ls='-',
#         label=rf'$r = {r}\ \mathrm{{m}}$',
#     )
#     for r, col in RANGE_COLORS.items()
# ]

# ax.legend(
#     handles=legend_handles,
#     loc='lower left',
#     framealpha=0.92,
#     facecolor='white',
#     edgecolor='0.7',
#     handlelength=2.3,
#     borderpad=0.65,
#     labelspacing=0.4,
#     handletextpad=0.55,
# )


# ── Layout and save ───────────────────────────────────────────────────────────
# Slightly more space around axes within the figure canvas.
fig.subplots_adjust(left=0.085, right=0.975, bottom=0.12, top=0.90)

fig.savefig(
    OUT,
    dpi=DPI,
    facecolor='white',
)

fig.savefig(
    OUT.replace('.png', '.pdf'),
    bbox_inches='tight',
    facecolor='white',
)

fig.savefig(
    OUT.replace('.png', '.svg'),
    bbox_inches='tight',
    facecolor='white',
)

print(f"Saved -> {OUT}  ({W}x{H} px)")
print(f"Saved -> {OUT.replace('.png', '.pdf')}")
print(f"Saved -> {OUT.replace('.png', '.svg')}")