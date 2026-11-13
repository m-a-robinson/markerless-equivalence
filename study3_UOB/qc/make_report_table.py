"""Per-participant summary table (markdown) from the QC outputs. Usage: python3 make_report_table.py > table.md"""
import csv
from collections import defaultdict

R = list(csv.DictReader(open('results/qc_issues.csv')))
C = list(csv.DictReader(open('results/stride_counts.csv')))
Rec = list(csv.DictReader(open('results/stride_recommendations.csv')))
cnt = {}
for r in C:
    if r['file'] in ('LKXV', 'RKXV', 'PEXV'):
        cnt.setdefault((r['subject'], r['condition']), {})[r['file'][0]] = (int(r['n_unique']), int(r['n_strides']))
rem = defaultdict(list)
for r in Rec:
    if r['recommendation'].startswith('REMOVE'):
        rem[(r['subject'], r['condition'])].append({'left': 'L', 'right': 'R', 'pelvis': 'P'}[r['side']] + r['stride'])
part = defaultdict(set)
for r in R:
    if r['severity'] == 'info' or not r['subject'] or r['check'] == 'stride':
        continue
    conds = ['WU', 'UOB'] if r['condition'] in ('WU vs UOB', 'UOB+WU') else [r['condition']]
    j = r['joint']
    if r['check'] == 'participant':
        lab = ('high variability: ' if 'within' in r['metric'] else 'mean outlier: ') + j
    elif r['check'] == 'duplicate':
        lab = ('**copied trial (both conds)**' if '+' in r['condition'] else
               f'duplicate strides ({j.replace("all ", "").replace(" joints", "")})')
    elif r['check'] == 'structure':
        lab = 'pelvis export mismatch' if 'Pelvis' in j else 'low stride count'
    else:
        lab = {'LR': 'L-R asymmetry: ', 'WU_UOB': 'WU≠UOB: ', 'identity': 'WU/UOB ID mismatch',
               'model': '**different model (knee rot)**'}[r['check']] + (j if r['check'] in ('LR', 'WU_UOB') else '')
    for cd in conds:
        part[(r['subject'], cd)].add(lab)
print('| Participant | Cond | Strides L, R, pelvis (unique/exported) | Strides to remove from all variables | '
      'Participant-level flags |')
print('|---|---|---|---|---|')
for k in sorted(cnt):
    c = cnt[k]
    s = ', '.join(f'{c[x][0]}/{c[x][1]}' if c[x][0] != c[x][1] else str(c[x][1]) for x in 'LRP')
    print(f'| {k[0]} | {k[1]} | {s} | {", ".join(rem[k]) or "–"} | {"; ".join(sorted(part[k])) or "–"} |')
