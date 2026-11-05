"""Squash trace of the final models (squash on) restricted to the steps the module acts on, with the start frames drawn from the second halves of both trajectories: 8 rollouts of 1000 steps from random MD
frames, random initial orientations, seed 7, recording the factor h per protein step and whether the step closes on the
nearest neighbour. Per checkpoint: counts and summed h per nearest-neighbour bin over closing steps, the share of all
protein steps with h < 0.99, and the share contracted (h < 0.99 on a closing step). One file per tag in results/traces/;
tags already there are skipped.
usage: trace_squash.py   (DEVICE sets the device, default cuda)"""
import os

import numpy as np
import torch

from common import SYSTEMS, picks, reference
from paths import TRACES, check_md
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint

EDGES = np.linspace(0.8, 3.0, 45)
DEVICE = os.environ.get('DEVICE', 'cuda')


def trace(model, md, box, core, P, n=8, steps=1000, seed=7):
    g = np.random.default_rng(seed)
    frames = np.concatenate([v for v in md.values()])
    x0 = torch.tensor(frames[g.choice(len(frames), n, replace=False)].reshape(n, 1, -1), dtype=torch.float32, device=DEVICE)
    torch.manual_seed(seed)
    model.squash_trace = []
    kw = {'initial_theta': 2 * np.pi * torch.rand(n, P, device=DEVICE)} if getattr(model, 'arms_enabled', False) else {}
    with torch.no_grad():
        out = model.predict_future(x0, steps=steps, return_unwrapped=False, **kw)
    h = torch.stack([t[0].reshape(n, P) for t in model.squash_trace], 1).float().cpu().numpy()       # [n, steps, P]
    model.squash_trace = None
    sw = (out[0] if isinstance(out, tuple) else out).float().cpu().numpy().reshape(n, -1, P, 3)[..., :2]
    pos = np.concatenate([x0.cpu().numpy().reshape(n, 1, P, 3)[..., :2], sw[:, :-1]], 1)          # frame of each step's h
    disp = sw - pos
    disp -= box * np.round(disp / box)
    d = pos[:, :, None, :, :] - pos[:, :, :, None, :]                                             # [n,T,i,j,2] j - i
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))
    r[..., np.arange(P), np.arange(P)] = np.inf
    j = r.argmin(-1)
    rmin = np.take_along_axis(r, j[..., None], -1)[..., 0]
    v = np.take_along_axis(d, j[..., None, None].repeat(2, -1), -2)[..., 0, :] / rmin[..., None]
    closing = (disp * v).sum(-1) > 0
    idx = np.digitize((rmin / core).ravel(), EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1) & closing.ravel()
    cnt = np.bincount(idx[ok], minlength=len(EDGES) - 1)
    hsum = np.bincount(idx[ok], weights=h.ravel()[ok], minlength=len(EDGES) - 1)
    return cnt, hsum, float((h < 0.99).mean()), float(((h < 0.99) & closing).mean()), float(closing.mean())


if __name__ == '__main__':
    check_md(first=True)
    torch.set_grad_enabled(False)
    os.makedirs(TRACES, exist_ok=True)
    for system in SYSTEMS:
        md = reference(system)[0]
        for p in picks(system, ('sq',)):
            f = os.path.join(TRACES, p['tag'] + '.npz')
            if os.path.exists(f):
                continue
            model, cfg = load_membrane_checkpoint(p['path'], DEVICE, p['overrides'])
            model.eval()
            box, P = float(model.box_size), int(cfg['n_particles'])
            assert model.squash_head is not None, p['tag']
            cnt, hsum, low, contracted, closing = trace(model, md, box, float(model.ev_core) * box, P)   # the module's own hard core
            np.savez(f, edges=EDGES, cnt=cnt, hsum=hsum, frac=low, frac_contracted=contracted, frac_closing=closing)
            print(f"{p['tag']}: h < 0.99 on {100 * low:.2f}% of protein steps, contracted {100 * contracted:.2f}%, "
                  f"closing steps {100 * closing:.1f}%", flush=True)
            del model
            torch.cuda.empty_cache()
    print('trace_squash done', flush=True)
