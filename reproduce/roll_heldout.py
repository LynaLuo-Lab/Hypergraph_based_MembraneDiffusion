"""Held-out rollouts of the half-trajectory models: for every checkpoint of common.picks (final with squash, final
with the head off, baseline), 32 rollouts from frame 0 of each SECOND half, as long as it, seed 3000, the depth scale of
the first-half files the model trained on. Saves the rollouts (x,y and node depth) to assets/rollouts/heldout/ and one row per
checkpoint to results/rows_half.jsonl: the training scorer against a reference built from the second halves (z=False, rc/core
from the run's config; its floors are the other second half), and the per-trajectory log-pair-distance score κ with its
signed parts, also over frames 3000+. A checkpoint whose tag (run, variant, epoch) is already in the rows with its
rollouts on disk is skipped, so a changed selection rescored only where the epoch moved.
usage: roll_heldout.py [7|14]   (default both; KINDS=lk,base restricts the variants; DEVICE sets the device, default cuda)"""
import json
import os
import sys

import numpy as np
import torch

from common import N, SEED, SYSTEMS, heldout_lk, picks, reference
from membrane import fit_objective as fo
from membrane.FitObjectiveCallback import depth_std, fit_rollouts
from paths import RESULTS, ROLLS, check_md
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint

H = heldout_lk()
ROWS = os.path.join(RESULTS, 'rows_half.jsonl')
DEVICE = os.environ.get('DEVICE', 'cuda')


def late_share(x, md, rnd, box, start=3000):
    lm = H.lnpd(md[::H.STRIDE], box).ravel()
    kr = H.gparts(lm, H.lnpd(rnd, box).ravel())[0]
    k, m, sp = H.gparts(lm, H.lnpd(x[:, start::H.STRIDE], box).ravel())
    return {'late_logpair_kl_share': k / kr, 'late_logpair_mean_share': m / kr, 'late_logpair_spread_share': sp / kr}


if __name__ == '__main__':
    check_md(first=True)
    torch.set_grad_enabled(False)
    os.makedirs(ROLLS, exist_ok=True)
    done = {r['tag'] for r in map(json.loads, open(ROWS))} if os.path.exists(ROWS) else set()
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    for system in SYSTEMS:
        if not SYSTEMS[system]['prefix'] == f'h{only}' and only:
            continue
        md, depth, thetas, box = reference(system)
        rnd = None
        for p in picks(system, tuple(os.environ.get('KINDS', 'sq,nosq,base,lk').split(','))):
            f = os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz")
            if p['tag'] in done and os.path.exists(f):
                continue
            model, cfg = load_membrane_checkpoint(p['path'], DEVICE, p['overrides'])
            model.eval()
            assert abs(float(model.box_size) - box) < 1e-6 and (model.squash_head is not None) == (p['variant'] == 'sq'), p['tag']
            if rnd is None:
                rnd = np.random.default_rng(0).random((8000, int(cfg['n_particles']), 2)) * box
            ref = fo.Reference(md, box, rc=float(cfg.get('fit_rc') or fo.RC), core=float(cfg.get('fit_core') or fo.CORE), depth=depth)
            rolls, dep = fit_rollouts(model, md, thetas, N, seed=SEED, depth_scale=depth_std(list(cfg['membrane_data_paths'])))
            np.savez_compressed(f, **{f'xy_{k}': np.asarray(rolls[k])[..., :2].astype(np.float32) for k in md},
                                **{f'node_depth_{k}': np.asarray(dep[k][0]).astype(np.float32) for k in md})
            sc = {k: float(v) for k, v in fo.score(rolls, ref, depth=dep, z=False).items()}
            kl = {k: {a: float(b) for a, b in {**H.kl_metrics(rolls[k], md[k], rnd, box), **late_share(rolls[k], md[k], rnd, box)}.items()}
                  for k in md}
            row = {**{k: p[k] for k in ('system', 'run', 'seed', 'epoch', 'n_epochs', 'variant', 'tag')}, 'rollout_seed': SEED, 'n': N,
                   'rand_big': ref.rand_big, **sc, 'kl': kl}
            with open(ROWS, 'a') as fh:
                fh.write(json.dumps(row) + '\n')
            print(f"{p['tag']}: overlap {sc['overlap']:.4f} clustering {sc['clustering']:.3f} depth {sc['depth']:.3f} | κ "
                  + ' '.join(f"{k[-8:]} {v['logpair_kl_share']:.3f}" for k, v in kl.items()), flush=True)
            del model, rolls, dep
            torch.cuda.empty_cache()
    print('roll_heldout done', flush=True)
