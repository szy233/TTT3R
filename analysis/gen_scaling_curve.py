"""Generate scaling curve figures for DDD3R paper.

Fig A: Relpose ATE vs sequence length (ScanNet + TUM, 2-panel hero figure)
Fig B: Video depth abs_rel vs sequence length (KITTI + Bonn, 2-panel)

Data from experiment_results.md (validated).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# ── Global NeurIPS style ──
NEURIPS_TEXTWIDTH = 5.5

def setup_style():
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

OUT_DIR = Path("/home/szy/research/TTT3R/paper/fig")

# Color palette
C_CUT3R  = '#8B7BB5'
C_TTT3R  = '#D4877F'
C_CONST  = '#E6AB02'
C_BRAKE  = '#5BAA5B'
C_DDD3R  = '#3182BD'

MARKERS = {
    'CUT3R': 'o',
    'TTT3R': 's',
    'Constant': 'D',
    'Brake': '^',
    'DDD3R': 'v',
}

# ═══════════════════════════════════════════════════════════════════════
# Data from experiment_results.md (validated scaling curve tables)
# ═══════════════════════════════════════════════════════════════════════

# ScanNet relpose scaling curve (21 points, 4 methods)
SCANNET_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

SCANNET_DATA = {
    'CUT3R':    [0.045, 0.109, 0.194, 0.336, 0.401, 0.486, 0.541, 0.573, 0.622, 0.667,
                 0.677, 0.685, 0.700, 0.723, 0.751, 0.758, 0.787, 0.790, 0.801, 0.817],
    'TTT3R':    [0.034, 0.072, 0.107, 0.144, 0.166, 0.193, 0.218, 0.237, 0.257, 0.277,
                 0.296, 0.311, 0.323, 0.343, 0.357, 0.370, 0.380, 0.388, 0.397, 0.406],
    'Constant': [0.033, 0.072, 0.105, 0.141, 0.149, 0.169, 0.181, 0.192, 0.199, 0.214,
                 0.222, 0.233, 0.237, 0.242, 0.256, 0.261, 0.268, 0.272, 0.278, 0.283],
    'Brake':    [0.036, 0.080, 0.117, 0.155, 0.166, 0.185, 0.191, 0.211, 0.229, 0.282,
                 0.236, 0.242, 0.237, 0.246, 0.259, 0.255, 0.262, 0.261, 0.260, 0.261],
}

# TUM relpose scaling curve (12 points, 5 methods)
TUM_FRAMES = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

TUM_DATA = {
    'CUT3R':    [0.023, 0.034, 0.043, 0.056, 0.084, 0.108, 0.135, 0.141, 0.154, 0.158, 0.163, 0.166],
    'TTT3R':    [0.014, 0.020, 0.028, 0.041, 0.046, 0.055, 0.064, 0.067, 0.077, 0.086, 0.091, 0.103],
    'Constant': [0.012, 0.017, 0.021, 0.027, 0.033, 0.037, 0.042, 0.050, 0.058, 0.061, 0.068, 0.079],
    'Brake':    [0.012, 0.016, None, 0.024, 0.031, 0.033, 0.037, 0.043, 0.049, 0.053, 0.057, 0.063],
    'DDD3R':    [0.012, 0.015, 0.019, 0.025, 0.032, 0.033, 0.037, 0.040, 0.045, 0.049, 0.053, 0.055],
}

# KITTI video depth scaling curve (10 points, 5 methods)
KITTI_VD_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

KITTI_VD_DATA = {
    'CUT3R':    [0.0969, 0.1119, 0.1080, 0.1079, 0.1097, 0.1132, 0.1161, 0.1169, 0.1187, 0.1190],
    'TTT3R':    [0.0924, 0.1005, 0.0961, 0.0964, 0.0981, 0.1008, 0.1034, 0.1045, 0.1065, 0.1074],
    'Constant': [0.0908, 0.0963, 0.0932, 0.0936, 0.0954, 0.0979, 0.1008, 0.1028, 0.1054, 0.1065],
    'Brake':    [0.0906, 0.0949, 0.0927, 0.0930, 0.0946, 0.0969, 0.0998, 0.1020, 0.1050, 0.1063],
    'DDD3R':    [0.0913, 0.0992, 0.0962, 0.0953, 0.0956, 0.0969, 0.0991, 0.1005, 0.1027, 0.1033],
}

# Bonn video depth scaling curve (10 points, 5 methods)
BONN_VD_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

BONN_VD_DATA = {
    'CUT3R':    [0.0749, 0.0704, 0.0704, 0.0679, 0.0738, 0.0859, 0.0882, 0.0867, 0.0836, 0.0819],
    'TTT3R':    [0.0700, 0.0610, 0.0629, 0.0634, 0.0640, 0.0750, 0.0749, 0.0747, 0.0733, 0.0720],
    'Constant': [0.0660, 0.0567, 0.0582, 0.0608, 0.0615, 0.0710, 0.0702, 0.0700, 0.0692, 0.0687],
    'Brake':    [0.0643, 0.0556, 0.0568, 0.0592, 0.0596, 0.0685, 0.0679, 0.0679, 0.0669, 0.0661],
    'DDD3R':    [0.0647, 0.0581, 0.0596, 0.0597, 0.0606, 0.0705, 0.0701, 0.0700, 0.0690, 0.0678],
}


def plot_scaling_panel(ax, frames, data, ylabel, title, show_legend=False):
    """Plot one scaling curve panel."""
    method_style = {
        'CUT3R':    (C_CUT3R, MARKERS['CUT3R'], '-'),
        'TTT3R':    (C_TTT3R, MARKERS['TTT3R'], '-'),
        'Constant': (C_CONST, MARKERS['Constant'], '--'),
        'Brake':    (C_BRAKE, MARKERS['Brake'], '-'),
        'DDD3R':    (C_DDD3R, MARKERS['DDD3R'], '-'),
    }

    for method, values in data.items():
        color, marker, ls = method_style[method]
        # Handle None values (missing data points)
        x = [f for f, v in zip(frames, values) if v is not None]
        y = [v for v in values if v is not None]
        ax.plot(x, y, ls, color=color, marker=marker, markersize=3.5,
                markerfacecolor='white', markeredgewidth=0.8,
                markeredgecolor=color, label=method, zorder=3)

    ax.set_xlabel('Sequence length (frames)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.4)

    if show_legend:
        ax.legend(frameon=True, framealpha=0.9, edgecolor='none',
                  loc='upper left', fontsize=6.5)


def fig_relpose_scaling():
    """Hero figure: Relpose ATE scaling curve for ScanNet + TUM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NEURIPS_TEXTWIDTH, 2.5))

    plot_scaling_panel(ax1, SCANNET_FRAMES, SCANNET_DATA,
                       'ATE RMSE (m) $\\downarrow$', 'ScanNet',
                       show_legend=True)

    plot_scaling_panel(ax2, TUM_FRAMES, TUM_DATA,
                       'ATE RMSE (m) $\\downarrow$', 'TUM',
                       show_legend=False)

    # Annotate growth rates (200f→1000f)
    # ScanNet right margin annotations
    ax1.annotate('2.4$\\times$', xy=(1000, 0.817), xytext=(5, 0),
                 textcoords='offset points', fontsize=5.5, color=C_CUT3R,
                 va='center')
    ax1.annotate('1.7$\\times$', xy=(1000, 0.261), xytext=(5, 0),
                 textcoords='offset points', fontsize=5.5, color=C_BRAKE,
                 va='center')

    # TUM right margin annotations
    ax2.annotate('3.0$\\times$', xy=(1000, 0.166), xytext=(5, 0),
                 textcoords='offset points', fontsize=5.5, color=C_CUT3R,
                 va='center')
    ax2.annotate('2.2$\\times$', xy=(1000, 0.055), xytext=(5, 0),
                 textcoords='offset points', fontsize=5.5, color=C_DDD3R,
                 va='center')

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / 'scaling_relpose.pdf')
    fig.savefig(OUT_DIR / 'scaling_relpose.png', dpi=200)
    print('Saved scaling_relpose.pdf')
    plt.close(fig)


def fig_vdepth_scaling():
    """Video depth scaling curve for KITTI + Bonn."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(NEURIPS_TEXTWIDTH, 2.5))

    plot_scaling_panel(ax1, KITTI_VD_FRAMES, KITTI_VD_DATA,
                       'Abs Rel $\\downarrow$', 'KITTI (Video Depth)',
                       show_legend=True)

    plot_scaling_panel(ax2, BONN_VD_FRAMES, BONN_VD_DATA,
                       'Abs Rel $\\downarrow$', 'Bonn (Video Depth)',
                       show_legend=False)

    # KITTI: annotate crossover at ~300f
    ax1.axvline(300, color='#999999', linestyle=':', linewidth=0.6, alpha=0.5)
    ax1.annotate('crossover', xy=(300, 0.0969), xytext=(350, 0.114),
                 fontsize=5.5, color='#666666',
                 arrowprops=dict(arrowstyle='->', color='#999999', lw=0.5))

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / 'scaling_vdepth.pdf')
    fig.savefig(OUT_DIR / 'scaling_vdepth.png', dpi=200)
    print('Saved scaling_vdepth.pdf')
    plt.close(fig)


if __name__ == '__main__':
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_relpose_scaling()
    fig_vdepth_scaling()
