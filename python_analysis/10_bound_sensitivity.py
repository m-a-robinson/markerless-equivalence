#!/usr/bin/env python3
"""Sensitivity of the equivalence outcome to the choice of equivalence bound: figure and table.

The 90 % simultaneous paired CI of the mean difference MB - ML (spm1d, as in ``02_equivalence_analysis.py``) does not depend on
the bound, so it is computed once per variable and then judged against four alternative tolerances:

* +/-2 deg                 McGinley et al. (2009) "clinically negligible" criterion;
* +/-5 deg                 primary bound of the manuscript (McGinley et al. 2009, upper limit of the 'reasonable' zone);
* markerless MDC           minimal detectable change of the markerless system (Augustine et al. 2025; Study 1 of the repository,
                           ``mdc_bounds``; foot progression, not examined, is kept at 5 deg);
* marker-based MDC         inter-session MDC of marker-based gait analysis in healthy adults (Horsak et al. 2026).  The values
                           were read from the published forest plot and are listed in ``MB_MDC`` below - check them against the paper.

Each node of the gait cycle is equivalent / inconclusive / not equivalent (same rule as the main analysis); results are the
percentage of the 101 nodes.  The same outlier selection as the other analyses is used (``outlier_selection.csv``).

Outputs: ``figures/bound_sensitivity.jpg`` (stacked bars, one panel per bound with +/-5 deg first and a dashed line before the alternatives, variables sorted by % equivalent at +/-5 deg; no mean row),
``results/table_bound_sensitivity.csv`` and ``.docx`` (bound in degrees and the three percentages for every scenario).

Usage
-----
    python 10_bound_sensitivity.py [--selection outlier_selection.csv] [--dpi 300]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (ANALYSED, EQUIVALENT, FIGURES_DIR, INCONCLUSIVE, NOT_EQUIVALENT, RESULTS_DIR, SELECTION_FILE,
                    classify_equivalence, load_dataset, load_selection, paired_ci, participant_means)

# Horsak et al. (2026), healthy adults, inter-session MDC95 (degrees), read from the forest plot - in the order of ANALYSED.
MB_MDC = dict(PelTilt=4.6, PelObl=3.7, PelRot=3.4, HipFlex=6.7, HipAdd=4.8, HipRot=15.8, KneeFlex=4.8, KneeVar=8.3,
              KneeRot=9.7, AnkDF=4.4, AnkInv=7.6, FootProg=7.6)
SCENARIOS = [('±5°', 'Primary'), ('±2°', 'McGinley'), ('ML MDC', 'Markerless MDC'), ('MB MDC', 'Marker-based MDC')]
COLOURS = {EQUIVALENT: '#2ca25f', INCONCLUSIVE: '#f0a020', NOT_EQUIVALENT: '#d62728'}   # as in the equivalence figure
ORDER = (EQUIVALENT, INCONCLUSIVE, NOT_EQUIVALENT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selection', type=Path, default=SELECTION_FILE)
    ap.add_argument('--alpha-ci', type=float, default=0.1)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--out', type=Path, default=FIGURES_DIR / 'bound_sensitivity.jpg')
    ap.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    ds, sel = load_dataset(), load_selection(args.selection)
    print(f'Outlier selection: {sel.describe()}')
    rows = []
    for k, var in enumerate(ANALYSED):
        mb, ml, subs = participant_means(ds, var.code, sel)
        lo, hi, _ = paired_ci(mb, ml, args.alpha_ci)
        bounds = [5.0, 2.0, float(ds.mdc_bounds[k]), MB_MDC[var.code]]
        r = dict(code=var.code, variable=var.title, n=len(subs))
        for (lab, _), b in zip(SCENARIOS, bounds):
            cls = classify_equivalence(lo, hi, b)
            r[lab] = {"bound": b, **{c: 100 * float(np.mean(cls == c)) for c in ORDER}}
        rows.append(r)
    rows.sort(key=lambda r: -r['±5°'][EQUIVALENT])
    mean = {lab: {c: float(np.mean([r[lab][c] for r in rows])) for c in ORDER} | {'bound': float('nan')} for lab, _ in SCENARIOS}

    # ---- table ----
    args.results_dir.mkdir(parents=True, exist_ok=True)
    hdr = ['Variable'] + [f'{lab}_{s}' for lab, _ in SCENARIOS for s in ('bound_deg', 'equiv_pct', 'inconcl_pct', 'notequiv_pct')]
    with open(args.results_dir / 'table_bound_sensitivity.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows:
            w.writerow([r['variable']] + [f"{r[lab]['bound']:.2f}" if s == 'b' else f"{r[lab][c]:.1f}"
                                          for lab, _ in SCENARIOS for s, c in (('b', 0), ('e', EQUIVALENT), ('i', INCONCLUSIVE), ('n', NOT_EQUIVALENT))])
        w.writerow(['Mean of the variables'] + [v for lab, _ in SCENARIOS for v in
                                                ('', f'{mean[lab][EQUIVALENT]:.1f}', f'{mean[lab][INCONCLUSIVE]:.1f}', f'{mean[lab][NOT_EQUIVALENT]:.1f}')])
    print(f"{'':22s}" + ''.join(f'{lab:>22s}' for lab, _ in SCENARIOS) + '   (bound; % E/I/NE)')
    for r in rows + [dict(variable='Mean', **mean)]:
        print(f"{r['variable']:22s}" + ''.join(
            f"{('' if np.isnan(r[lab]['bound']) else format(r[lab]['bound'], '5.1f')):>5s} {r[lab][EQUIVALENT]:4.0f}/{r[lab][INCONCLUSIVE]:3.0f}/{r[lab][NOT_EQUIVALENT]:3.0f}   "
            for lab, _ in SCENARIOS))

    try:
        import docx
        from docx.enum.section import WD_ORIENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt
        d = docx.Document()
        sec = d.sections[0]; sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
        t = d.add_table(rows=2, cols=1 + 4 * len(SCENARIOS)); t.style = 'Table Grid'
        h0, h1 = t.rows[0].cells, t.rows[1].cells
        h0[0].merge(h1[0]); h0[0].text = 'Variable'
        for g, (lab, _) in enumerate(SCENARIOS):
            j = 1 + 4 * g
            h0[j].merge(h0[j + 3]); h0[j].text = lab
            for jj, s in zip(range(j, j + 4), ('Bound (°)', 'Equiv.', 'Inconcl.', 'Not equiv.')):
                h1[jj].text = s
        for r in rows + [dict(variable='Mean of the variables', **mean)]:
            cells = t.add_row().cells
            cells[0].text = r['variable']
            for g, (lab, _) in enumerate(SCENARIOS):
                v = r[lab]
                for jj, s in enumerate((v['bound'], v[EQUIVALENT], v[INCONCLUSIVE], v[NOT_EQUIVALENT])):
                    cells[1 + 4 * g + jj].text = '' if np.isnan(s) else (f'{s:.1f}' if jj == 0 else f'{s:.0f}')
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(8); run.bold = ri < 2 or row.cells[0].text.startswith('Mean of')
        note = d.add_paragraph(
            'Percentage of the gait cycle (101 points) that is equivalent, inconclusive or not equivalent for the 90% simultaneous '
            'CI of the mean difference (marker-based − markerless) judged against alternative bounds: ±2° and ±5° (McGinley et al. 2009), '
            'the markerless minimal detectable change (ML MDC; Augustine et al. 2025; foot progression ±5°) and the marker-based '
            'inter-session MDC (MB MDC; Horsak et al. 2026). Percentages are rounded.')
        for run in note.runs:
            run.font.size = Pt(8)
        d.save(args.results_dir / 'table_bound_sensitivity.docx')
    except ImportError:
        print('python-docx not installed - table written as .csv only')

    # ---- figure ----
    fig, axes = plt.subplots(1, 4, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.975, top=0.84, bottom=0.1, wspace=0.3)
    y = np.arange(len(rows))[::-1]
    names = [r['variable'] for r in rows]
    for ax, (lab, sub) in zip(axes, SCENARIOS):
        for yi, r in zip(y, rows):
            left = 0
            for c in ORDER:
                ax.barh(yi, r[lab][c], left=left, color=COLOURS[c], height=0.72, lw=0)
                if r[lab][c] >= 12:
                    ax.text(left + r[lab][c] / 2, yi, f'{r[lab][c]:.0f}', ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
                left += r[lab][c]
            if not np.isnan(r[lab]['bound']):
                ax.text(101, yi, f"{r[lab]['bound']:.1f}°", ha='left', va='center', fontsize=6.5, color='0.3')
        ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.tick_params(axis="y", length=0); ax.tick_params(labelsize=7.5)
        ax.set_title(f'{lab}\n{sub}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Gait cycle (%)', fontsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    x0 = 0.5 * (axes[0].get_position().x1 + axes[1].get_position().x0)      # line separating the primary bound from the alternatives
    fig.add_artist(Line2D([x0, x0], [0.08, 0.87], color='0.3', lw=1.2, ls='--', transform=fig.transFigure))
    fig.legend(handles=[Patch(color=COLOURS[c], label=l) for c, l in zip(ORDER, ('Equivalent', 'Inconclusive', 'Not equivalent'))],
               loc='upper center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.55, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
