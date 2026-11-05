"""Where the 7-PIEZO forecast's first-window excess at long pair distance (180-230 nm) and at positive node depth (5-15 nm)
comes from: share of pair distances / node depths in those ranges for MD frame 0 of each second half, MD's first 2,000
frames, all MD frames, and the model's first and last 2,000-frame windows (40k rollouts, as forecast.py). Writes
results/bump_half.json and prints it.
usage: forecast_bump.py"""
import json
import os

import numpy as np

from common import SYSTEMS, forecast_picks, reference
from membrane import fit_objective as fo
from paths import RESULTS, ROLLS40K

STRIDE, WIN, NW, K = 4, 2000, 20, 500
name = '7-PIEZO'
s = SYSTEMS[name]
md, depth, _, box = reference(name)
nm = s['nm']
share = lambda x, lo, hi: float(((x >= lo) & (x <= hi)).mean())
pd = lambda x: fo.pair_dist_xy(x, box).ravel() * nm
out = {}
for k, v in md.items():
    d = depth[k]['node'] * nm
    fin = lambda a: a[np.isfinite(a)]
    out[k] = {'md_frame0': {'pair_180_230': share(pd(v[:1]), 180, 230), 'depth_5_15': share(fin(d[:1]), 5, 15), 'depth_frame0_finite': bool(np.isfinite(d[0]).all())},
              'md_first2000': {'pair_180_230': share(pd(v[:WIN:STRIDE]), 180, 230), 'depth_5_15': share(fin(d[:WIN]), 5, 15)},
              'md_all': {'pair_180_230': share(pd(v[::STRIDE]), 180, 230), 'depth_5_15': share(fin(d), 5, 15)}}
    first, last = {'pair': [], 'dep': []}, {'pair': [], 'dep': []}
    for p in forecast_picks(name):
        z = np.load(os.path.join(ROLLS40K, f"{p['tag']}_{k}.npz"))
        xy, nd = z['xy'][:, 1:], z['node_depth'] * nm
        for tgt, sl in ((first, slice(0, K)), (last, slice((NW - 1) * K, NW * K))):
            tgt['pair'].append(share(pd(xy[:, sl]), 180, 230))
            tgt['dep'].append(share(nd[:, sl], 5, 15))
    out[k]['model_first2000'] = {'pair_180_230': float(np.mean(first['pair'])), 'depth_5_15': float(np.mean(first['dep']))}
    out[k]['model_last2000'] = {'pair_180_230': float(np.mean(last['pair'])), 'depth_5_15': float(np.mean(last['dep']))}
json.dump(out, open(os.path.join(RESULTS, 'bump_half.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
