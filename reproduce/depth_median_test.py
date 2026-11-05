"""per-run median node depth, full model vs backbone and comparison models, two-sided Mann-Whitney across seeds
usage: depth_median_test.py"""
import glob
import os

import numpy as np
from scipy.stats import mannwhitneyu

from paths import ROLLS

G = {'full': 'h7_lk_both_sq_s70?_sq_e*_seed3000.npz', 'backbone': 'h7_lk_both_s70?_lk_e*_seed3000.npz', 'comparison': 'h7_base_s70?_base_e*_seed3000.npz'}
K = ('7piezo_R1second', '7piezo_R2second')
V = {m: {k: [float(np.median(np.load(f)[f'node_depth_{k}'])) for f in sorted(glob.glob(os.path.join(ROLLS, g)))] for k in K} for m, g in G.items()}
for k in K:
    for b in ('backbone', 'comparison'):
        print(k, 'full vs', b, 'n', len(V['full'][k]), len(V[b][k]), 'p', round(mannwhitneyu(V['full'][k], V[b][k], alternative='two-sided').pvalue, 3))
