"""Forecast check: the final models with squash (common.forecast_picks: 7-PIEZO the six with most epochs, 14-PIEZO all), ROLLS
rollouts (default 32) of 40,000 frames from frame 0 of each second half (rollout seed 3000, so the first frames repeat
roll_heldout.py's rollouts). Keeps every 4th frame of x,y and node depth (npz units) in assets/rollouts/forecast/<tag>_<trajectory>.npz;
files already there are skipped.
usage: roll_forecast.py [7|14]   (ROLLS=n sets the rollouts per trajectory; DEVICE sets the device, default cuda)"""
import os
import sys
import time

import numpy as np
import torch

from common import SEED, SYSTEMS, forecast_picks, reference
from membrane.FitObjectiveCallback import depth_std
from paths import ROLLS40K, check_md
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint

OUT = ROLLS40K
DEVICE = os.environ.get('DEVICE', 'cuda')
NR, STEPS, STRIDE = int(os.environ.get('ROLLS', 32)), 40000, 4

if __name__ == '__main__':
    check_md(first=True)
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    for system in SYSTEMS:
        if only and SYSTEMS[system]['prefix'] != f'h{only}':
            continue
        md, _, thetas, box = reference(system)
        for p in forecast_picks(system):
            if all(os.path.exists(os.path.join(OUT, f"{p['tag']}_{k}.npz")) for k in md):
                continue
            model, cfg = load_membrane_checkpoint(p['path'], DEVICE, p['overrides'])
            model.eval()
            assert model.squash_head is not None, p['tag']
            P, scale = int(cfg['n_particles']), depth_std(list(cfg['membrane_data_paths']))
            for k, v in md.items():
                f = os.path.join(OUT, f"{p['tag']}_{k}.npz")
                if os.path.exists(f):
                    continue
                t0 = time.time()
                x0 = torch.remainder(torch.tensor(v[0].reshape(1, 1, -1), dtype=torch.float32, device=DEVICE), box).repeat(NR, 1, 1)
                kw = {'initial_theta': torch.tensor(thetas[k], device=DEVICE).unsqueeze(0).repeat(NR, 1)} \
                    if getattr(model, 'arms_enabled', False) and k in thetas else {}
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                sw, mn, mh = model.predict_future(x0, steps=STEPS, return_unwrapped=False, return_membrane=True, **kw)
                xy = torch.cat([x0.float(), sw.float()], 1).reshape(NR, STEPS + 1, P, 3)[:, ::STRIDE, :, :2]
                nd = mn.float()[:, STRIDE - 1::STRIDE] * scale[0]          # depth at frames STRIDE, 2*STRIDE, ...
                np.savez_compressed(f, xy=xy.cpu().numpy().astype(np.float32), node_depth=nd.cpu().numpy().astype(np.float32),
                                    stride=STRIDE, box=box, md_len=len(v), seed=SEED, rolls=NR)
                del sw, mn, mh, xy, nd
                torch.cuda.empty_cache()
                print(f"{p['tag']} {k}: {STEPS} frames x {NR} rollouts in {time.time() - t0:.0f} s", flush=True)
            del model
            torch.cuda.empty_cache()
    print('roll_forecast done', flush=True)
