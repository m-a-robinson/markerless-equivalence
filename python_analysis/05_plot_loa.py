#!/usr/bin/env python3
"""Figure: pointwise and simultaneous limits of agreement between markerless and marker-based kinematics.

Reads ``results/loa_<tag>.npz`` written by ``03_loa_analysis.py`` and draws one panel per variable:

* thin grey lines   - each participant's difference curve MB - ML;
* black line        - mean difference;
* dashed black      - pointwise functional LoA, mean +/- 1.96 SD (Roislien et al. 2012);
* light-blue band   - simultaneous 95 % prediction band, mean +/- z* SD sqrt(1+1/n) (spm1d random-field theory);
* green dashed      - the +/-5 degree equivalence bounds;
* red double arrow  - width of the pointwise LoA where it is largest (degrees);
* green shading     - where the agreement is ACCEPTABLE: light green where the pointwise LoA lie inside the bounds,
                      darker green where the simultaneous band does too (a new participant's whole curve is then
                      expected inside).  Nowhere else is a verdict implied: amber and red are reserved for the
                      equivalence outcomes (``04_plot_equivalence.py``).

The text in each panel gives the mean width of both bands and the percentage of the gait cycle for which the
band lies inside the bounds.

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

from common import FIGURES_DIR, RESULTS_DIR, runs

GREEN, RED, BAND = '#33ee33', '#b00020', '#d6e6f5'
OK_PW, OK_SIM = '#8fd1ae', '#2f9e63'        # acceptable agreement: pointwise LoA inside / simultaneous band inside


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

    fig, axes = plt.subplots(4, 3, figsize=(8, 9.8), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.905, bottom=0.06, hspace=0.3, wspace=0.22)
    for ax, code, title in zip(axes.ravel(), meta['variables'], meta['titles']):
        d, md = z[f'{code}_d'], z[f'{code}_md']
        lo, hi, slo, shi = z[f'{code}_lo'], z[f'{code}_hi'], z[f'{code}_slo'], z[f'{code}_shi']
        w = hi - lo

        ax.fill_between(x, slo, shi, color=BAND, lw=0, zorder=1)
        for mask, col, alpha in (((lo >= -bound) & (hi <= bound), OK_PW, 0.40),
                                 ((slo >= -bound) & (shi <= bound), OK_SIM, 0.30)):   # acceptable agreement
            for s0, e0 in runs(mask):
                ax.axvspan(s0 - 0.5, e0 + 0.5, color=col, alpha=alpha, lw=0, zorder=2)
        ax.plot(x, d.T, color='0.55', lw=0.5, alpha=0.6, zorder=3)          # drawn above the band
        ax.plot(x, lo, '--', color='k', lw=1.2, zorder=4)
        ax.plot(x, hi, '--', color='k', lw=1.2, zorder=4)
        ax.plot(x, md, 'k', lw=2.2, zorder=5)
        ax.axhline(0, color='k', lw=0.5, ls=':')
        for s in (bound, -bound):
            ax.axhline(s, color=GREEN, ls='--', lw=1.6, zorder=6)

        i = int(np.argmax(w))                                    # widest point of the pointwise LoA
        ax.annotate('', xy=(i, lo[i]), xytext=(i, hi[i]), zorder=7,
                    arrowprops=dict(arrowstyle='<->', color=RED, lw=1.4, shrinkA=0, shrinkB=0))
        ax.text(i + (3 if i < 70 else -3), md[i], f'{w[i]:.0f}°', color=RED, fontsize=7.5, fontweight='bold',
                ha='left' if i < 70 else 'right', va='center', zorder=7,
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))

        in_p = 100 * np.mean((lo >= -bound) & (hi <= bound))
        in_s = 100 * np.mean((slo >= -bound) & (shi <= bound))
        ax.text(0.02, 0.97, f'width {w.mean():.1f}° pointwise, {np.mean(shi - slo):.1f}° simult.\n'
                            f'within ±{bound:g}°: {in_p:.0f}% / {in_s:.0f}%',
                transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color='#333')
        lim = max(10, 5 * np.ceil(max(np.abs(slo).max(), np.abs(shi).max(), bound) / 5))
        ax.set_xlim(0, 100); ax.set_ylim(-lim, lim * 1.25)
        ax.set_title(f'{title} (n={meta["n"][meta["variables"].index(code)]})', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('MB − ML (°)', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Gait Cycle (%)', fontsize=8)

    fig.legend(handles=[Line2D([0], [0], color='k', lw=2, label='Mean difference'),
                        Line2D([0], [0], color='k', ls='--', lw=1.2, label='Pointwise LoA (± 1.96 SD)'),
                        Patch(color=BAND, label='Simultaneous 95% prediction band (RFT)'),
                        Line2D([0], [0], color=GREEN, ls='--', lw=1.6, label=f'±{bound:g}° bounds'),
                        Line2D([0], [0], color='0.55', lw=1, label='Individual participants'),
                        Line2D([0], [0], color=RED, marker='$\\leftrightarrow$', ls='', ms=9,
                               label='Max pointwise LoA width'),
                        Patch(color=OK_PW, alpha=0.6, label=f'Pointwise LoA inside ±{bound:g}° (acceptable)'),
                        Patch(color=OK_SIM, alpha=0.5, label=f'Simultaneous band also inside ±{bound:g}°')],
               loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.997))
    out = args.out or FIGURES_DIR / f'loa_{args.tag}.jpg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
