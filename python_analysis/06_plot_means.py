#!/usr/bin/env python3
"""Figure: grand mean +/- 1 SD of marker-based (MB) and markerless (ML) lower-limb kinematics.

Python version of the manuscript means figure (``mlmb_mean_sds_apr2026.jpg``, produced in MATLAB with
``spm1d.plot.plot_meanSD``).  One panel per variable on a 4 x 3 grid:

* thick line     - mean across participants of the participant mean curves (red = markerless, blue = marker-based);
* shaded band    - +/- 1 SD across participants (sample SD, N - 1);
* thin grey line - zero;
* y-limits       - the same fixed limits as the manuscript figure (they differ by variable).

The participant means are built with the current outlier selection (``outlier_selection.csv``, see
``select_outliers.py``), exactly as in the equivalence and LoA analyses, so excluded strides and participants do not
enter the means.  The legend sits above the panels.

Unlike the MATLAB figure, whose x-data start at 1 on an axis labelled from 0, nodes here are plotted at their true
gait-cycle position (0-100 %).

Usage
-----
    python 06_plot_means.py [--selection outlier_selection.csv] [--out figures/means_sd.jpg] [--dpi 300]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import ANALYSED, FIGURES_DIR, SELECTION_FILE, load_dataset, load_selection, participant_means

COLOURS = {'ml': '#ff0000', 'mb': '#0000ff'}              # as in the manuscript figure
# y-limits of the manuscript figure, in the order of ``ANALYSED`` (pelvis, hip, knee, foot, ankle)
YLIM_MIN = [-10, -15, -30, -30, -30, -30, -10, -30, -30, -30, -30, -30]
YLIM_MAX = [30, 15, 30, 60, 30, 30, 70, 30, 30, 30, 30, 30]
BAND_ALPHA = 0.25


def mean_sd_plot(ax, curves, colour, x):
    """Mean line and +/- 1 SD band of ``curves`` (participants x nodes)."""
    m, sd = curves.mean(0), curves.std(0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=colour, alpha=BAND_ALPHA, lw=0)
    ax.plot(x, m, color=colour, lw=2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selection', type=Path, default=SELECTION_FILE, help='outlier selection file')
    ap.add_argument('--out', type=Path, default=FIGURES_DIR / 'means_sd.jpg', help='output image')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    ds = load_dataset()
    sel = load_selection(args.selection)
    print(f'Outlier selection: {sel.describe()}')
    x = np.arange(101)

    fig, axes = plt.subplots(4, 3, figsize=(6, 6.9))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.07, hspace=0.42, wspace=0.3)
    for k, (ax, var) in enumerate(zip(axes.ravel(), ANALYSED)):
        mb, ml, subs = participant_means(ds, var.code, sel)
        mean_sd_plot(ax, mb, COLOURS['mb'], x)
        mean_sd_plot(ax, ml, COLOURS['ml'], x)
        ax.axhline(0, color='0.35', lw=0.7, zorder=0)
        ax.set_title(var.title, fontsize=9, fontweight='bold', pad=4)
        ax.set_xlim(0, 100); ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(YLIM_MIN[k], YLIM_MAX[k])
        ax.tick_params(labelsize=7.5)
        for sp in ('top', 'right'):                          # "box off"
            ax.spines[sp].set_visible(False)
        if k % 3 == 0:
            ax.set_ylabel('Angle (°)', fontsize=8.5)
        if k >= 9:
            ax.set_xlabel('Gait Cycle (%)', fontsize=8.5)
        if len(subs) != ds.mb.shape[0]:
            print(f'  {var.title}: n = {len(subs)}')

    fig.legend(handles=[Line2D([0], [0], color=COLOURS['ml'], lw=2.5, label='Markerless'),
                        Line2D([0], [0], color=COLOURS['mb'], lw=2.5, label='Marker-based')],
               loc='upper center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
