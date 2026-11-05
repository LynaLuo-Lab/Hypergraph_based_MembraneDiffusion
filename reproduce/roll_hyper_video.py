"""Re-runs the held-out rollout batch of the best full-model run on second-half trajectory 2 exactly as roll_heldout.py does (same
checkpoint, overrides, seed, depth scale) to get the hyperedge depth the saved rolls lack, checks it against the saved rollouts,
and keeps the chosen rollout (xy, node depth, hyper depth in npz units) in assets/videos/<system>_traj2_hyper.npz.
usage: roll_hyper_video.py [7|14]   (DEVICE sets the device, default cuda)"""
import os
import sys

import numpy as np
import torch

from common import N, SEED, SYSTEMS, reference
from figures import setup
from membrane.FitObjectiveCallback import depth_std, fit_rollouts
from paths import HYPER, ROLLS, check_md
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint
from video import JOBS, pick

DEVICE = os.environ.get('DEVICE', 'cuda')

check_md(first=True)
torch.set_grad_enabled(False)
os.makedirs(HYPER, exist_ok=True)
for k in sys.argv[1:] or JOBS:
    name, key, f = JOBS[k]
    s = setup(name)
    s['name'] = name
    x_saved, d_saved, p, j = pick(s, key)
    model, cfg = load_membrane_checkpoint(p['path'], DEVICE, p['overrides'])
    model.eval()
    npz = np.load([q for q in SYSTEMS[name]['npz'] if key in q][0], allow_pickle=True)
    inc = [(n, b) for n, b in model.named_buffers() if n.endswith('H') and b.dim() == 2 and tuple(b.shape) == tuple(npz['members'].shape)]
    assert inc, 'no incidence buffer'
    same = [bool((b.cpu().numpy().astype(bool) == npz['members'].astype(bool)).all()) for _, b in inc]
    print(f'{name}: incidence buffers {[n for n, _ in inc]} equal to npz members: {same}', flush=True)
    assert all(same)
    md = {key: s['md'][key]}
    _, _, thetas, _ = reference(name)
    rolls, dep = fit_rollouts(model, md, thetas, N, seed=SEED, depth_scale=depth_std(list(cfg['membrane_data_paths'])))
    x, (dn, dh) = rolls[key][..., :2].astype(np.float32), dep[key]
    z = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
    dx = np.abs(x - z['xy_' + key]).max()
    dd = np.abs(dn.astype(np.float32) - z['node_depth_' + key]).max()
    print(f'{name}: max |xy rerun - saved| (all 32) {dx:.3g}; max |node depth rerun - saved| {dd:.3g}; chosen rollout {j} xy diff {np.abs(x[j] - x_saved).max():.3g}', flush=True)
    if dx != 0 or dd != 0:
        print('MISMATCH: not writing', flush=True)
        continue
    np.savez_compressed(os.path.join(HYPER, f + '_hyper.npz'), xy=x[j], node=dn[j].astype(np.float32), hyper=dh[j].astype(np.float32),
                        members=npz['members'], rollout=j, tag=p['tag'])
    print('wrote hyper npz', dh[j].shape, flush=True)
    del model
    torch.cuda.empty_cache()
