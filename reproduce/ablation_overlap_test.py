"""overlap of the ablation (full-model checkpoints, squash off) against the backbone model, 7-PIEZO, two-sided Mann-Whitney across seeds
usage: ablation_overlap_test.py"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

from common import picks
from paths import RESULTS

rows = {r['tag']: r for r in map(json.loads, open(os.path.join(RESULTS, 'rows_half.jsonl')))}
ov = {v: [rows[p['tag']]['overlap'] for p in picks('7-PIEZO', (v,))] for v in ('nosq', 'lk')}
print('overlap ablation', len(ov['nosq']), f"{min(ov['nosq']):.4f}-{max(ov['nosq']):.4f}", 'backbone', len(ov['lk']), f"{min(ov['lk']):.4f}-{max(ov['lk']):.4f}")
print('mean ratio', np.mean(ov['nosq']) / np.mean(ov['lk']))
print('Mann-Whitney p', mannwhitneyu(ov['nosq'], ov['lk'], alternative='two-sided').pvalue)
