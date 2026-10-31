#!/usr/bin/env python3
"""Figure: functional limits of agreement (LoA) between markerless and marker-based kinematics, judged against the bounds.

Reads ``results/loa_<tag>.npz`` written by ``03_loa_analysis.py`` and draws one panel per variable:

* thin grey lines   - each participant's difference curve MB - ML;
* black line        - mean difference;
* dashed black      - pointwise functional LoA, mean +/- 1.96 SD (Roislien et al. 2012);
* green dashed      - the +/-5 degree bounds;
* red double arrow  - width of the LoA where it is largest (degrees);
* shaded region     - the area between the LoA, coloured at each point of the gait cycle by ``common.classify_loa``:

    green  the LoA lie inside the bounds                           -> acceptable agreement for individuals
    amber  the LoA exceed a bound but the mean difference is inside -> agreement at group level only
    red    the mean difference itself is beyond a bound              -> systematic bias

* bar under each panel - the same outcome at every point of the gait cycle (solid colours), laid out like the SPM bar
  of the equivalence figure.

The text in each panel gives the mean LoA width and the percentage of the gait cycle in each outcome.

Usage
-----
    python 05_plot_loa.py [--tag 5deg] [--dpi 300] [--out figures/loa_5deg.jpg]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (FIGURES_DIR, LOA_ACCEPTABLE, LOA_BIAS, LOA_GROUP_ONLY, RESULTS_DIR, classify_loa, runs)

BOUND_C, ARROW_C = '#33ee33', '#b00020'
FILL = {LOA_ACCEPTABLE: '#a6dcbd', LOA_GROUP_ONLY: '#f7d58b', LOA_BIAS: '#f08f82'}     # light fill of the LoA region
SOLID = {LOA_ACCEPTABLE: '#2ca25f', LOA_GROUP_ONLY: '#f0a020', LOA_BIAS: '#d62728'}    # solid colours of the outcome bar


def grow(mask):
    """Widen a node mask by one node on each side so neighbouring colour segments join without gaps."""
    return mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='5deg', help='tag of the results file (default 5deg)')
    ap.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    ap.add_argument('--out', type=Path, default=None, help='output image (default figures/loa_<tag>.jpg)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    z = np.load(args.results_dir / f'loa_{args.tag}.npz', allow_pickle=False)
    meta = json.loads(str(z['meta']))
    bound = meta['bound']
    x = np.arange(101)

    # layout knobs (same as 04_plot_equivalence.py)
    BAR_RATIO = 0.6          # outcome bar height relative to the panel (panel = 10)
    GAP_PANEL_BAR = 0.15     # gap between a panel and its own outcome bar
    GAP_BETWEEN_ROWS = 0.25  # gap between an outcome bar and the title of the next row

    fig = plt.figure(figsize=(8, 10.4))
    outer = fig.add_gridspec(4, 3, hspace=GAP_BETWEEN_ROWS, wspace=0.22, left=0.08, right=0.98, top=0.905, bottom=0.06)
    for k, (code, title) in enumerate(zip(meta['variables'], meta['titles'])):
        r, c = divmod(k, 3)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[10, BAR_RATIO], hspace=GAP_PANEL_BAR)
        ax, axb = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        d, md, lo, hi = z[f'{code}_d'], z[f'{code}_md'], z[f'{code}_lo'], z[f'{code}_hi']
        w = hi - lo
        cls = classify_loa(md, lo, hi, bound)

        for cc in (LOA_ACCEPTABLE, LOA_GROUP_ONLY, LOA_BIAS):
            ax.fill_between(x, lo, hi, where=grow(cls == cc), color=FILL[cc], lw=0, zorder=1)
        ax.plot(x, d.T, color='0.55', lw=0.5, alpha=0.6, zorder=3)
        ax.plot(x, lo, '--', color='k', lw=1.2, zorder=4)
        ax.plot(x, hi, '--', color='k', lw=1.2, zorder=4)
        ax.plot(x, md, 'k', lw=2.2, zorder=5)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        for s in (bound, -bound):
            ax.axhline(s, color=BOUND_C, ls='--', lw=1.6, zorder=6)

        i = int(np.argmax(w))                                    # widest point of the LoA
        ax.annotate('', xy=(i, lo[i]), xytext=(i, hi[i]), zorder=7,
                    arrowprops=dict(arrowstyle='<->', color=ARROW_C, lw=1.4, shrinkA=0, shrinkB=0))
        ax.text(i + (3 if i < 70 else -3), md[i], f'{w[i]:.0f}°', color=ARROW_C, fontsize=7.5, fontweight='bold',
                ha='left' if i < 70 else 'right', va='center', zorder=7,
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
        pc = [100 * np.mean(cls == cc) for cc in (LOA_ACCEPTABLE, LOA_GROUP_ONLY, LOA_BIAS)]
        ax.text(0.02, 0.97, f'width {w.mean():.1f}° (mean)\ngreen/amber/red: {pc[0]:.0f}/{pc[1]:.0f}/{pc[2]:.0f}%',
                transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color='#333',
                bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.75), zorder=8)
        lim = max(10, 5 * np.ceil(max(np.abs(lo).max(), np.abs(hi).max(), bound) / 5))
        ax.set_xlim(0, 100); ax.set_ylim(-lim, lim * 1.25); ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_title(f'{title} (n={meta["n"][k]})', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7, labelbottom=False)
        if c == 0:
            ax.set_ylabel('MB − ML (°)', fontsize=8)

        # outcome bar (like the SPM bar in the equivalence figure)
        axb.set_xlim(0, 100); axb.set_ylim(0, 1); axb.set_yticks([])
        for cc in (LOA_ACCEPTABLE, LOA_GROUP_ONLY, LOA_BIAS):
            for s0, e0 in runs(cls == cc):
                axb.axvspan(s0 - 0.5, e0 + 0.5, color=SOLID[cc], lw=0)
        for sp in ('top', 'right', 'left'):
            axb.spines[sp].set_visible(False)
        axb.set_xticks([0, 20, 40, 60, 80, 100])
        axb.tick_params(labelsize=7, length=2, labelbottom=(r == 3))
        if r == 3:
            axb.set_xlabel('Gait Cycle (%)', fontsize=8)
        if c == 0:
            axb.set_ylabel('LoA', fontsize=7, rotation=0, ha='right', va='center')

    fig.legend(handles=[Patch(color=FILL[LOA_ACCEPTABLE], label=f'LoA inside ±{bound:g}°: acceptable'),
                        Patch(color=FILL[LOA_GROUP_ONLY], label=f'LoA exceed ±{bound:g}°, mean inside'),
                        Patch(color=FILL[LOA_BIAS], label=f'Mean difference beyond ±{bound:g}°'),
                        Line2D([0], [0], color='k', lw=2, label='Mean difference'),
                        Line2D([0], [0], color='k', ls='--', lw=1.2, label='Pointwise LoA (± 1.96 SD)'),
                        Line2D([0], [0], color=BOUND_C, ls='--', lw=1.6, label=f'±{bound:g}° bounds'),
                        Line2D([0], [0], color='0.55', lw=1, label='Individual participants'),
                        Line2D([0], [0], color=ARROW_C, marker='$\\leftrightarrow$', ls='', ms=9, label='Max LoA width')],
               loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.997))
    out = args.out or FIGURES_DIR / f'loa_{args.tag}.jpg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
