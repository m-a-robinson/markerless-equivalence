#!/usr/bin/env python3
"""Figure: equivalence of markerless and marker-based kinematics with SPM paired t-test.

Reads ``results/equivalence_<tag>.npz`` written by ``02_equivalence_analysis.py`` and draws one panel per variable:

* the mean difference MB - ML (black line) with its 90 % simultaneous CI (shaded band);
* the CI band coloured by outcome - green = equivalent (CI inside the bounds), amber = inconclusive (CI crosses a
  bound), red = not equivalent (CI entirely beyond a bound); the part of the band that extends past a bound is
  drawn darker;
* the equivalence bounds (green dashed lines) and the percentage of the gait cycle that is equivalent;
* a bar under each panel: black where the spm1d paired t-test (p < 0.05) indicates a statistical difference.

Usage
-----
    python 04_plot_equivalence.py [--tag 5deg] [--dpi 300] [--out figures/equivalence_5deg.jpg]
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

from common import (EQUIVALENT, FIGURES_DIR, INCONCLUSIVE, NOT_EQUIVALENT, RESULTS_DIR, runs)

C_BASE = '#d9d9d9'
C_BAND = {EQUIVALENT: '#a6dcbd', INCONCLUSIVE: '#f7d58b', NOT_EQUIVALENT: '#f08f82'}   # light: whole CI band
C_BEYOND = {INCONCLUSIVE: '#f0a020', NOT_EQUIVALENT: '#d62728'}                           # dark: part beyond a bound
C_BOUND = '#33ee33'


def grow(mask):
    """Widen a node mask by one node on each side so adjacent colour segments join without gaps."""
    return mask | np.r_[mask[1:], False] | np.r_[False, mask[:-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='5deg', help='tag of the results file (default 5deg)')
    ap.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    ap.add_argument('--out', type=Path, default=None, help='output image (default figures/equivalence_<tag>.jpg)')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    z = np.load(args.results_dir / f'equivalence_{args.tag}.npz', allow_pickle=False)
    meta = json.loads(str(z['meta']))
    x = np.arange(101)
    fixed = len(set(meta['bounds'])) == 1
    bound_label = f"±{meta['bounds'][0]:g}° bounds" if fixed else 'MDC bounds'

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(8, 3, height_ratios=[10, 1.1] * 4, hspace=0.35, wspace=0.22,
                          left=0.08, right=0.98, top=0.93, bottom=0.06)
    for k, (code, title, b) in enumerate(zip(meta['variables'], meta['titles'], meta['bounds'])):
        r, c = divmod(k, 3)
        ax, axb = fig.add_subplot(gs[2 * r, c]), fig.add_subplot(gs[2 * r + 1, c])
        lo, hi, md = z[f'{code}_lo'], z[f'{code}_hi'], z[f'{code}_md']
        cls, sig = z[f'{code}_cls'], z[f'{code}_spm_sig']

        # CI band coloured by outcome, darker where it extends beyond a bound
        ax.fill_between(x, lo, hi, color=C_BASE, lw=0)
        for cc in (INCONCLUSIVE, EQUIVALENT, NOT_EQUIVALENT):
            ax.fill_between(x, lo, hi, where=grow(cls == cc), color=C_BAND[cc], lw=0)
        for cc in (INCONCLUSIVE, NOT_EQUIVALENT):
            m = grow(cls == cc)
            ax.fill_between(x, np.maximum(lo, b), hi, where=m & (hi > b), color=C_BEYOND[cc], lw=0)
            ax.fill_between(x, lo, np.minimum(hi, -b), where=m & (lo < -b), color=C_BEYOND[cc], lw=0)
        ax.plot(x, md, 'k', lw=2.2)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        for s in (b, -b):
            ax.axhline(s, color=C_BOUND, ls='--', lw=1.6)
        ymax = max(10, 5 * np.ceil(max(np.abs(lo).max(), np.abs(hi).max(), b) / 5))
        ax.set_xlim(0, 100); ax.set_ylim(-ymax, ymax + 5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.text(0.98, 0.97, f'{100 * np.mean(cls == EQUIVALENT):.0f}% Equiv.', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#1fbf3a', fontweight='bold')
        ax.tick_params(labelsize=7, labelbottom=False)
        if c == 0:
            ax.set_ylabel('Angle (°)', fontsize=8)

        # statistical difference (spm1d paired t-test)
        axb.set_ylim(0, 1); axb.set_yticks([])
        for s0, e0 in runs(sig):
            axb.axvspan(s0 - 0.5, e0 + 0.5, color='k', lw=0)
        for sp in ('top', 'right', 'left'):
            axb.spines[sp].set_visible(False)
        axb.set_xlim(0, 100); axb.set_xticks([0, 20, 40, 60, 80, 100])
        axb.tick_params(labelsize=7, length=2, labelbottom=(r == 3))
        if r == 3:
            axb.set_xlabel('Gait Cycle (%)', fontsize=8)
        if c == 0:
            axb.set_ylabel('SPM', fontsize=7, rotation=0, ha='right', va='center')

    fig.legend(handles=[Patch(color=C_BAND[EQUIVALENT], label='Equivalent (CI inside bounds)'),
                        Patch(color=C_BEYOND[NOT_EQUIVALENT], label='Not equivalent (CI beyond bound)'),
                        Patch(color=C_BEYOND[INCONCLUSIVE], label='Inconclusive (CI crosses bound)'),
                        Line2D([0], [0], color=C_BOUND, ls='--', label=bound_label),
                        Patch(color='k', label='SPM paired t-test p<0.05')],
               loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    out = args.out or FIGURES_DIR / f'equivalence_{args.tag}.jpg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
