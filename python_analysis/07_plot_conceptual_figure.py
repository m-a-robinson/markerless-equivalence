"""Conceptual figure: possible outcomes of the equivalence test (group level) and of the limits-of-agreement
analysis (individual level).  Synthetic data only - this is a schematic, not a result.

Panels
  a  one gait-cycle point: how a 90% CI relates to the equivalence bounds (and to a significance test)
  b  across the gait cycle: CI band coloured by outcome, SPM bar underneath
  c  one gait-cycle point: distribution of individual differences, LoA vs the bounds
  d  across the gait cycle: individual curves, pointwise LoA, simultaneous band, bounds
"""
#!/usr/bin/env python3
"""Conceptual figure: possible outcomes of the equivalence test (group level) and of the limits-of-agreement (LoA)
analysis (individual level).

All data are synthetic and chosen only to illustrate the decision rules; nothing here is a result of the study.
The equivalence bound is drawn as the symbol Delta (the study bound, for example the MDC or +/-5 deg, is not needed to
read the figure); the synthetic data are tuned for a bound of 5 units.

Panels
------
a  One point of the gait cycle: five example 90 % CIs of the mean difference MB - ML against the bounds +/-Delta,
   with the outcome (equivalent / inconclusive / not equivalent) and what a paired significance test (SPM) would say.
b  Across the gait cycle: the mean difference and its CI band coloured by outcome (the part of the band beyond a bound
   drawn darker), the share of the cycle that is equivalent, and an SPM bar underneath.
c  One point of the gait cycle: distribution of individual differences with the LoA (mean +/- 1.96 SD) against the
   bounds - acceptable (LoA inside +/-Delta) or not acceptable (LoA exceed +/-Delta), the latter either with the group
   mean inside the bounds or with a systematic bias.
d  Across the gait cycle: smooth individual difference curves, pointwise LoA, the simultaneous prediction band
   (a schematic multiplier of 3.0 stands in for the random-field-theory threshold) and the bounds, with bars showing
   where each band lies inside +/-Delta.

Colours match the results figures.  Equivalence outcomes (panels a, b): green = equivalent, amber = inconclusive,
red = not equivalent.  Agreement (panels c, d): green marks ACCEPTABLE agreement (LoA inside +/-Delta; the lighter green is
the pointwise LoA, the darker green also the simultaneous band) and the simultaneous band itself is light blue, so no
verdict is implied where the LoA exceed the bounds.  The equivalence classification uses the same function as the analysis
(``common.classify_equivalence``).

Usage
-----
    python 07_plot_conceptual_figure.py [--out figures/conceptual_outcomes.png] [--dpi 200] [--seed 7]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (EQUIVALENT, FIGURES_DIR, INCONCLUSIVE, NOT_EQUIVALENT, classify_equivalence, runs)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out', type=Path, default=FIGURES_DIR / 'conceptual_outcomes.png', help='output image')
ap.add_argument('--dpi', type=int, default=200)
ap.add_argument('--seed', type=int, default=7, help='seed of the synthetic individual curves in panel d')
args = ap.parse_args()
OUT = args.out
OUT.parent.mkdir(parents=True, exist_ok=True)

D = 5.0                                                     # equivalence bound (schematic, "Delta")
C_EQ, C_INC, C_NE = '#2ca25f', '#f0a020', '#d62728'          # strong colours: equivalent / inconclusive / not equivalent
L_EQ, L_INC, L_NE = '#a6dcbd', '#f7d58b', '#f08f82'          # light band colours
C_BOUND, C_BAND = '#33ee33', '#d6e6f5'                       # bounds; simultaneous prediction band (light blue, as in 05)
OK_PW, OK_SIM = '#8fd1ae', '#2f9e63'                         # acceptable agreement: pointwise LoA / simultaneous band inside
C_NOTOK = '#6b6b6b'                                          # LoA exceed the bounds (no verdict colour)
x = np.arange(101)
rng = np.random.default_rng(args.seed)

fig = plt.figure(figsize=(12.5, 10.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1.08], left=0.135, right=0.985, top=0.875,
                      bottom=0.135, hspace=0.36, wspace=0.34)


def panel_letter(ax, s):
    ax.text(-0.02, 1.13, s, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')


def bounds_lines(ax, horizontal=False):
    f = ax.axhline if horizontal else ax.axvline
    for s in (D, -D):
        f(s, color=C_BOUND, ls='--', lw=1.8, zorder=2)


# ------------------------------------------------------------------ a: decision rule at one point
ax = fig.add_subplot(gs[0, 0])
rows = [  # label, mean, lo, hi, class, SPM significant?
    ('CI inside bounds,\nincludes 0', 0.4, -2.6, 3.4, EQUIVALENT, False),
    ('CI inside bounds,\nexcludes 0', 2.4, 1.0, 3.8, EQUIVALENT, True),
    ('CI crosses\none bound', 3.6, 1.2, 6.4, INCONCLUSIVE, True),
    ('CI crosses\nboth bounds', 0.6, -7.2, 8.4, INCONCLUSIVE, False),
    ('CI entirely\nbeyond a bound', 8.2, 6.4, 10.0, NOT_EQUIVALENT, True),
]
col = {EQUIVALENT: C_EQ, INCONCLUSIVE: C_INC, NOT_EQUIVALENT: C_NE}
name = {EQUIVALENT: 'Equivalent', INCONCLUSIVE: 'Inconclusive', NOT_EQUIVALENT: 'Not equivalent'}
ax.axvspan(-D, D, color=L_EQ, alpha=0.35, lw=0)
bounds_lines(ax)
ax.axvline(0, color='k', lw=0.8, ls=':')
for i, (lab, m, lo, hi, c, sig) in enumerate(rows):
    y = len(rows) - i
    ax.plot([lo, hi], [y, y], color=col[c], lw=3, solid_capstyle='butt')
    ax.plot([lo, lo], [y - 0.14, y + 0.14], color=col[c], lw=2); ax.plot([hi, hi], [y - 0.14, y + 0.14], color=col[c], lw=2)
    ax.plot(m, y, 's', color='k', ms=6, zorder=5)
    ax.text(-13.2, y, lab, ha='right', va='center', fontsize=8.5)
    ax.annotate(name[c], xy=(1.03, y), xycoords=('axes fraction', 'data'), fontsize=9.5, fontweight='bold',
                color=col[c], va='center')
    ax.annotate('SPM: ' + ('different' if sig else 'n.s.'), xy=(1.03, y - 0.3), xycoords=('axes fraction', 'data'),
                fontsize=7.5, color='0.35', va='center')
ax.set_xlim(-12, 12); ax.set_ylim(0.3, len(rows) + 0.85)
ax.set_yticks([]); ax.set_xticks([-10, -D, 0, D, 10]); ax.set_xticklabels(['−10', '−Δ', '0', '+Δ', '10'])
ax.set_xlabel('Mean difference MB − ML (°)', fontsize=9)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.set_title('Equivalence test at one point of the gait cycle\n(90% CI of the mean difference vs ±Δ)', fontsize=10, fontweight='bold', pad=10)
ax.text(0, len(rows) + 0.62, 'equivalence region', ha='center', fontsize=8, color='#1b7a45')
axa = ax

# ------------------------------------------------------------------ b: equivalence across the cycle
subb = gs[0, 1].subgridspec(2, 1, height_ratios=[10, 0.7], hspace=0.12)
axb = fig.add_subplot(subb[0])
md = 1.0 + 9.6 * np.exp(-((x - 66) / 12) ** 2) - 1.0 * np.sin(x / 10.0)
hw = 1.6 + 1.5 * np.exp(-((x - 38) / 13) ** 2) + 1.4 * np.exp(-((x - 95) / 7) ** 2) + 0.4 * np.exp(-((x - 66) / 12) ** 2)
lo, hi = md - hw, md + hw
cls = classify_equivalence(lo, hi, D)


def grow(m):
    return m | np.r_[m[1:], False] | np.r_[False, m[:-1]]


axb.fill_between(x, lo, hi, color='#d9d9d9', lw=0)
for cc, lc in ((INCONCLUSIVE, L_INC), (EQUIVALENT, L_EQ), (NOT_EQUIVALENT, L_NE)):
    axb.fill_between(x, lo, hi, where=grow(cls == cc), color=lc, lw=0)
for cc, dc in ((INCONCLUSIVE, C_INC), (NOT_EQUIVALENT, C_NE)):
    m = grow(cls == cc)
    axb.fill_between(x, np.maximum(lo, D), hi, where=m & (hi > D), color=dc, lw=0)
    axb.fill_between(x, lo, np.minimum(hi, -D), where=m & (lo < -D), color=dc, lw=0)
axb.plot(x, md, 'k', lw=2.4)
axb.axhline(0, color='k', lw=0.6, ls=':')
bounds_lines(axb, horizontal=True)
axb.set_xlim(0, 100); axb.set_ylim(-9.5, 21.8)
axb.set_xticks([]); axb.set_ylabel('Mean difference MB − ML (°)', fontsize=9)
axb.text(1.5, D + 0.5, '+Δ', color='#1b7a45', fontsize=9, fontweight='bold'); axb.text(1.5, -D - 1.7, '−Δ', color='#1b7a45', fontsize=9, fontweight='bold')
labpos = {EQUIVALENT: (None, 17.6), INCONCLUSIVE: (None, 17.6), NOT_EQUIVALENT: (None, 17.6)}
for cc in (EQUIVALENT, INCONCLUSIVE, NOT_EQUIVALENT):
    r = runs(cls == cc)
    if r:
        s0, e0 = max(r, key=lambda t: t[1] - t[0])
        axb.plot([s0, e0], [16.3, 16.3], color=col[cc], lw=3.5, solid_capstyle='butt')
        axb.text((s0 + e0) / 2 + (6 if cc == NOT_EQUIVALENT else 0), 17.7 + (2.0 if cc == NOT_EQUIVALENT else 0), name[cc], ha='center', va='bottom', fontsize=8.6, fontweight='bold', color=col[cc])
axb.text(99, -8.6, f'{100 * np.mean(cls == EQUIVALENT):.0f}% of the cycle equivalent', ha='right', fontsize=8.5, color='#1b7a45', fontweight='bold')
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
axb.set_title('Equivalence across the gait cycle (group level)', fontsize=10, fontweight='bold', pad=14)
sig = (lo > 0) | (hi < 0)
bar = fig.add_subplot(subb[1])
bar.set_xlim(0, 100); bar.set_ylim(0, 1); bar.set_yticks([])
for s0, e0 in runs(sig):
    bar.axvspan(s0 - 0.5, e0 + 0.5, color='k', lw=0)
bar.set_xticks([0, 20, 40, 60, 80, 100]); bar.tick_params(labelsize=8)
for sp in ('top', 'right', 'left'):
    bar.spines[sp].set_visible(False)
bar.set_xlabel('Gait cycle (%)', fontsize=9)
bar.text(-0.015, 0.5, 'SPM', transform=bar.transAxes, ha='right', va='center', fontsize=7.5)

# ------------------------------------------------------------------ c: LoA at one point
sc = [  # mean, sd, outcome text, colour
    (0.6, 1.1, 'Acceptable agreement (individual level)', OK_SIM, 'LoA inside ±Δ: individual differences are expected to be acceptable'),
    (0.8, 3.5, 'Not acceptable: LoA exceed ±Δ', C_NOTOK, 'mean inside ±Δ (group means equivalent), but individual differences are too large'),
    (6.6, 1.5, 'Not acceptable: LoA exceed ±Δ', C_NOTOK, 'the mean difference itself exceeds ±Δ (systematic bias)'),
]
sub = gs[1, 0].subgridspec(3, 1, hspace=0.55)
xs = np.linspace(-14, 14, 400)
for k, (m, s_, ttl, c, expl) in enumerate(sc):
    a = fig.add_subplot(sub[k])
    dens = np.exp(-0.5 * ((xs - m) / s_) ** 2) / (s_ * np.sqrt(2 * np.pi))
    a.fill_between(xs, dens, color=c, alpha=0.28, lw=0)
    a.plot(xs, dens, color=c, lw=1.5)
    FR = 0.58                                                    # fraction of the axes used by the drawing
    a.axvspan(-D, D, ymax=FR, color=L_EQ, alpha=0.35, lw=0)
    for b_ in (D, -D):
        a.axvline(b_, ymax=FR, color=C_BOUND, ls='--', lw=1.8)
    a.axvline(m, ymax=FR, color='k', lw=1.4)
    top = dens.max() * 1.18
    a.annotate('', xy=(m - 1.96 * s_, top), xytext=(m + 1.96 * s_, top),
               arrowprops=dict(arrowstyle='|-|', color='k', lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=4))
    a.text(m + 1.96 * s_ + 0.4, top, 'LoA', fontsize=8, va='center')
    a.set_xlim(-14, 14); a.set_ylim(0, top / FR * 1.02); a.set_yticks([])
    for sp in ('top', 'right', 'left'):
        a.spines[sp].set_visible(False)
    a.set_xticks([-10, -D, 0, D, 10] if k == 2 else []); a.set_xticklabels(['−10', '−Δ', '0', '+Δ', '10'] if k == 2 else [])
    if k == 2:
        a.set_xlabel('Individual differences MB − ML (°)', fontsize=9)
    a.text(-13.6, 0.985, ttl, transform=a.get_xaxis_transform(), fontsize=9.5, fontweight='bold', color=c, va='top', ha='left')
    a.text(-13.6, 0.815, expl, transform=a.get_xaxis_transform(), fontsize=7.6, color='0.3', va='top', ha='left')
    if k == 0:
        a.set_title('Limits of agreement at one point of the gait cycle\n(distribution of individual differences; LoA = mean ± 1.96 SD)',
                    fontsize=10, fontweight='bold', pad=14, loc='left')
axc0 = fig.axes[-3]

# ------------------------------------------------------------------ d: LoA across the cycle
subd = gs[1, 1].subgridspec(3, 1, height_ratios=[10, 0.5, 0.5], hspace=0.16)
axd = fig.add_subplot(subd[0])
n = 20
t01 = x / 100.0
# mean MB-ML difference: a slowly varying offset with a gait-like swing-phase excursion
mean_t = 1.0 + 0.9 * np.sin(2 * np.pi * (t01 - 0.05)) - 1.6 * np.exp(-((x - 67) / 9) ** 2)
# between-participant SD: small in stance, larger around pre-swing / swing, rising again at the end of the cycle
sd_t = 0.95 + 2.7 * np.exp(-((x - 68) / 14) ** 2) + 0.9 * np.exp(-((x - 100) / 9) ** 2)
# smooth participant deviations: a constant offset plus a few low-frequency harmonics (at most 4 cycles per stride)
harm = np.array([[rng.normal(0, 1 / k) * np.sin(2 * np.pi * k * t01 + rng.uniform(0, 2 * np.pi)) for k in range(1, 5)]
                 for _ in range(n)]).sum(1)                                   # n x 101, smooth
std = lambda v: (v - v.mean(0)) / v.std(0, ddof=1)
off = rng.standard_normal(n); off = (off - off.mean()) / off.std(ddof=1)
curves = mean_t + sd_t * (0.78 * off[:, None] + 0.62 * std(harm))           # sample SD equals sd_t at every node
m_s, sd_s = curves.mean(0), curves.std(0, ddof=1)
plo, phi = m_s - 1.96 * sd_s, m_s + 1.96 * sd_s
zstar = 3.0                                                       # schematic random-field-theory threshold
slo, shi = m_s - zstar * sd_s * np.sqrt(1 + 1 / n), m_s + zstar * sd_s * np.sqrt(1 + 1 / n)
axd.fill_between(x, slo, shi, color=C_BAND, lw=0, zorder=1)
for mask_, col_, alpha_ in (((plo >= -D) & (phi <= D), OK_PW, 0.40), ((slo >= -D) & (shi <= D), OK_SIM, 0.30)):
    for s0_, e0_ in runs(mask_):
        axd.axvspan(s0_ - 0.5, e0_ + 0.5, color=col_, alpha=alpha_, lw=0, zorder=2)      # acceptable agreement
axd.plot(x, curves.T, color='0.55', lw=0.6, alpha=0.7, zorder=3)
axd.plot(x, plo, 'k--', lw=1.3, zorder=4); axd.plot(x, phi, 'k--', lw=1.3, zorder=4)
axd.plot(x, m_s, 'k', lw=2.4, zorder=5)
axd.axhline(0, color='k', lw=0.6, ls=':')
bounds_lines(axd, horizontal=True)
axd.set_xlim(0, 100); axd.set_ylim(-14.5, 14.5); axd.set_xticks([])
axd.set_ylabel('Difference MB − ML (°)', fontsize=9)
axd.text(1.5, D + 0.6, '+Δ', color='#1b7a45', fontsize=9, fontweight='bold'); axd.text(1.5, -D - 2.0, '−Δ', color='#1b7a45', fontsize=9, fontweight='bold')
for sp in ('top', 'right'):
    axd.spines[sp].set_visible(False)
axd.set_title('Limits of agreement across the gait cycle (individual level)', fontsize=10, fontweight='bold', pad=14)
in_p = (plo >= -D) & (phi <= D)
in_s = (slo >= -D) & (shi <= D)
axd.text(99, 13.8, "Simultaneous band is wider than the pointwise LoA:\nit covers a new participant's whole curve with 95% probability", ha='right', va='top',
         fontsize=7.8, color='#2c5f8f')
for j, (mask, label, c) in enumerate(((in_p, 'Pointwise LoA\ninside ±Δ', OK_PW), (in_s, 'Simultaneous band\ninside ±Δ', OK_SIM))):
    b2 = fig.add_subplot(subd[1 + j])
    b2.set_xlim(0, 100); b2.set_ylim(0, 1); b2.set_yticks([])
    for sp in ('top', 'right', 'left'):
        b2.spines[sp].set_visible(False)
    for s0, e0 in runs(mask):
        b2.axvspan(s0 - 0.5, e0 + 0.5, color=c, lw=0)
    b2.text(-0.015, 0.5, label, transform=b2.transAxes, ha='right', va='center', fontsize=7.3, linespacing=0.95)
    if j == 0:
        b2.set_xticks([]); b2.spines['bottom'].set_visible(False)
    else:
        b2.set_xticks([0, 20, 40, 60, 80, 100]); b2.tick_params(labelsize=8)
        b2.set_xlabel('Gait cycle (%)', fontsize=9)

# ------------------------------------------------------------------ legends / reading guide
fig.legend(handles=[Patch(color=L_EQ, label='Equivalent: CI inside ±Δ'), Patch(color=C_INC, label='Inconclusive: CI crosses a bound'),
                    Patch(color=C_NE, label='Not equivalent: CI beyond a bound'),
                    Line2D([0], [0], color=C_BOUND, ls='--', lw=1.8, label='Bounds ±Δ (e.g. MDC or ±5°)'),
                    Line2D([0], [0], color='k', lw=2, label='Mean difference'),
                    Line2D([0], [0], color='k', ls='--', lw=1.3, label='Pointwise LoA (±1.96 SD)'),
                    Patch(color=C_BAND, label='Simultaneous prediction band (RFT)'),
                    Line2D([0], [0], color='0.55', lw=1, label='Individual participants'),
                    Patch(color=OK_PW, alpha=0.7, label='LoA inside ±Δ: acceptable'),
                    Patch(color=OK_SIM, alpha=0.7, label='Simultaneous band also inside ±Δ')],
           loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.004))
fig.text(0.5, 0.975, 'Group equivalence and individual agreement: possible outcomes (schematic, synthetic data)',
         ha='center', fontsize=13, fontweight='bold')
# panel letters: one baseline per row, one left edge per column (figure coordinates)
row1 = max(axa.get_position().y1, axb.get_position().y1) + 0.045
row2 = max(axd.get_position().y1, axc0.get_position().y1) + 0.045
col1 = axa.get_position().x0 - 0.012
col2 = axb.get_position().x0 - 0.012
for letter, xx, yy in (('a', col1, row1), ('b', col2, row1), ('c', col1, row2), ('d', col2, row2)):
    fig.text(xx, yy, letter, fontsize=16, fontweight='bold', ha='right', va='bottom')
fig.savefig(OUT, dpi=args.dpi)
assert set(cls.tolist()) == {EQUIVALENT, INCONCLUSIVE, NOT_EQUIVALENT}, 'panel b must show all three outcomes'
print(f'Wrote {OUT}  (panel d: pointwise LoA inside the bounds for {100 * in_p.mean():.0f}% of the cycle, '
      f'simultaneous band for {100 * in_s.mean():.0f}%; synthetic data)')
