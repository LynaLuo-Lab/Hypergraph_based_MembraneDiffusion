"""Numbers the text derives from the result files: overlap ratios and overlapping pairs per frame, the share of MD's clustering above random
placement, the node-depth offset as a share of MD's 5-95% range, and the forecast length over the second-half length.
usage: derived_numbers.py"""
import json
import os

import numpy as np

from common import SYSTEMS, picks
from paths import RESULTS

rows = {r['tag']: r for r in map(json.loads, open(os.path.join(RESULTS, 'rows_half.jsonl')))}
st = json.load(open(os.path.join(RESULTS, 'stats_half.json')))
fc = json.load(open(os.path.join(RESULTS, 'forecast_stats_half.json')))
NPAIRS = {'7-PIEZO': 21, '14-PIEZO': 91}
VN = {'sq': 'full', 'nosq': 'ablation', 'lk': 'backbone', 'base': 'comparison'}
STEPS = 40000

for name in SYSTEMS:
    s, f = st[name], fc[name]
    print(f'== {name}')
    ov = {v: float(np.mean([rows[p['tag']]['overlap'] for p in picks(name, (v,))])) for v in VN}
    for v in VN:
        print(f'overlap {VN[v]}: {ov[v]:.4f}; pairs per frame {ov[v] * NPAIRS[name]:.3f} (one per {1 / (ov[v] * NPAIRS[name]):.1f} frames)')
    for a, b in (('nosq', 'sq'), ('nosq', 'lk'), ('base', 'sq'), ('base', 'lk'), ('lk', 'sq'), ('base', 'nosq')):
        print(f'overlap ratio {VN[a]} / {VN[b]}: {ov[a] / ov[b]:.2f}')
    rnd = s['random_mean_largest_cluster']
    for k, md in s['md_mean_largest_cluster'].items():
        print(k, 'share of MD clustering above random (model - random) / (MD - random), %:',
              {VN[v]: round(100 * (s['model_mean_largest_cluster_pooled'][v][k] - rnd) / (md - rnd), 1) for v in VN})
        q = f[k]['md_node_depth_q5_50_95_nm_all']
        print(k, 'node-depth median offset from MD as % of MD 5-95% range', f'({q[2] - q[0]:.1f} nm):',
              {VN[v]: round(100 * abs(s['model_node_depth_median_nm'][v][k] - s['md_node_depth_median_nm'][k]) / (q[2] - q[0]), 1) for v in VN})
        print(k, f'forecast length / second-half length: {STEPS / s["md_len"][k]:.2f} ({STEPS} / {s["md_len"][k]})')
