import matplotlib.pyplot as plt

BLUE   = '#0072B2'
ORANGE = '#D55E00'


GRAY       = '#808080'
GRAY_EDGE  = '#4D4D4D'
GRAY_FILL  = '#999999'
GRAY_LIGHT = '#cccccc'

LEGEND_KW = dict(frameon=True, fontsize=12, edgecolor='#e0e0e0',
                 fancybox=True, facecolor='white')

LABEL_KW = dict(fontsize=13, labelpad=11)


def setup_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'


def apply_style(ax):
    ax.grid(True, linestyle='-', color='gray', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6)
    ax.tick_params(axis='both', which='minor', direction='out', length=3)


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)