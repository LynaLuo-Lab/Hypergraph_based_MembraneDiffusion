"""Start-up transient of the junction (hyperedge) depth in the full model's held-out rollouts (common.picks(system, ('sq',))), both
second-half trajectories. Re-runs fit_rollouts as roll_hyper_video.py does (same checkpoint, overrides, seed, depth scale), 32 rollouts,
but on MD cut to its first 2001 frames, so 2000 rollout frames; the xy is checked against the saved held-out rollouts. Depth in nm.
Rollout frame f (1..2000) is MD frame f. Per-edge frame-1 difference: median over rollouts of the model, minus MD, median over edges.
Half-time: first frame at which the model's median junction depth (median over rollouts and hyperedges) has covered half of the
gap between its frame-1 value and its mean over frames 1500-2000. Writes junction_transient.txt / .json.
usage: junction_transient.py [N_RUNS_LIMIT]   (DEVICE sets the device, default cuda)"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import N, SEED, SYSTEMS, picks, reference
from membrane.FitObjectiveCallback import depth_std, fit_rollouts
from paths import RESULTS, ROLLS
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint

torch.set_grad_enabled(False)
FR = [1, 10, 100, 500, 1000, 2000]
T = 2000
lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
dev = os.environ.get('DEVICE', 'cuda')
print(f'device {dev}', flush=True)

res, t_run = [], []
for name in SYSTEMS:
    nm = SYSTEMS[name]['nm']
    md, depth, thetas, box = reference(name)
    for p in picks(name, ('sq',))[:lim]:
        t0 = time.time()
        model, cfg = load_membrane_checkpoint(p['path'], dev, p['overrides'])
        model.eval()
        mm = model.membrane_module
        a_h, a_n = torch.sigmoid(mm.ou_a_logit_hyper).cpu().numpy(), torch.sigmoid(mm.ou_a_logit_node).cpu().numpy()
        scale = depth_std(list(cfg['membrane_data_paths']))
        cut = {k: v[:T + 1] for k, v in md.items()}
        rolls, dep = fit_rollouts(model, cut, thetas, N, seed=SEED, depth_scale=scale)
        saved = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
        for key in md:
            x = rolls[key][..., :2].astype(np.float32)
            dx = float(np.abs(x - saved['xy_' + key][:, :T + 1]).max())
            dn, dh = dep[key][0] * nm, dep[key][1] * nm                    # [N, T, P], [N, T, H]
            hm, nm_md = depth[key]['hyper'] * nm, depth[key]['node'] * nm
            cm = np.median(dh, axis=(0, 2))                                 # model median junction depth per frame
            tail = cm[1499:2000].mean()
            gap = tail - cm[0]
            hit = np.nonzero((cm - cm[0]) * np.sign(gap) >= abs(gap) / 2)[0]
            ok = np.isfinite(hm).all(-1)
            res.append(dict(system=name, traj=key, run=p['run'], seed=p['seed'], epoch=p['epoch'], xy_maxdiff=dx,
                            frames=FR,
                            model_hyper=[float(np.median(dh[:, f - 1])) for f in FR], model_node=[float(np.median(dn[:, f - 1])) for f in FR],
                            md_hyper=[float(np.median(hm[f])) for f in FR], md_node=[float(np.median(nm_md[f])) for f in FR],
                            md_hyper_all=float(np.median(hm[ok])), md_node_all=float(np.median(nm_md[ok])),
                            diff_f1=float(np.median(np.median(dh[:, 0], axis=0) - hm[1])),
                            a_hyper_med=float(np.median(a_h)), a_hyper_rng=[float(a_h.min()), float(a_h.max())],
                            a_node_med=float(np.median(a_n)), a_node_rng=[float(a_n.min()), float(a_n.max())],
                            inv_a_hyper=float(1 / np.median(a_h)), inv_a_node=float(1 / np.median(a_n)),
                            cm_f1=float(cm[0]), cm_tail=float(tail), half_time=int(hit[0] + 1) if len(hit) else None,
                            half_time_ou=float(np.log(2) / np.median(a_h))))
        t_run.append(time.time() - t0)
        print(f"{p['run']} done in {t_run[-1]:.0f} s; max xy diff {[r['xy_maxdiff'] for r in res[-2:]]}", flush=True)
        del model, rolls, dep
        if dev == 'cuda':
            torch.cuda.empty_cache()

def ms(v, d=1):
    v = np.array(v)
    return f'{v.mean():+.{d}f} [{v.min():+.{d}f},{v.max():+.{d}f}]'

L = [f'device {dev}; runs {len(t_run)}, mean {np.mean(t_run):.0f} s/run, total {sum(t_run):.0f} s; depth in nm; mean [min,max] over runs',
     f'max |xy rerun - saved| over all runs, rollouts, trajectories, first 2001 frames: {max(r["xy_maxdiff"] for r in res):.3g}', '']
summ = {}
for name in SYSTEMS:
    for key in sorted({r['traj'] for r in res if r['system'] == name}):
        rr = [r for r in res if r['system'] == name and r['traj'] == key]
        if not rr:
            continue
        L.append(f'{name} {key} ({len(rr)} runs); frames {FR}')
        s = {}
        for lab in ('model_hyper', 'md_hyper', 'model_node', 'md_node'):
            L.append(f'  {lab:12s} ' + ' | '.join(ms([r[lab][i] for r in rr]) for i in range(len(FR))))
            s[lab] = [[float(np.mean([r[lab][i] for r in rr])), float(min(r[lab][i] for r in rr)), float(max(r[lab][i] for r in rr))] for i in range(len(FR))]
        L.append(f'  MD over all 2nd-half frames: junction {rr[0]["md_hyper_all"]:+.1f}, node {rr[0]["md_node_all"]:+.1f}')
        L.append(f'  frame-1 junction diff (model-MD, median over edges): {ms([r["diff_f1"] for r in rr])}')
        L.append(f'  a_hyper median {ms([r["a_hyper_med"] for r in rr], 5)}, 1/a {ms([r["inv_a_hyper"] for r in rr])} frames; a_node median {ms([r["a_node_med"] for r in rr], 5)}, 1/a {ms([r["inv_a_node"] for r in rr])}')
        L.append(f'  half-time (frames) empirical {[r["half_time"] for r in rr]}; ln2/a_hyper {[round(r["half_time_ou"]) for r in rr]}')
        L.append(f'  model median junction frame 1 -> tail(1500-2000): {[(round(r["cm_f1"], 1), round(r["cm_tail"], 1)) for r in rr]}')
        summ[f'{name}|{key}'] = s
L.append('')
L.append('per run: system traj run a_hyper(med,min,max) a_node(med,min,max)')
for r in res:
    L.append(f'{r["system"]} {r["traj"]} {r["run"]} ah {r["a_hyper_med"]:.5f} {r["a_hyper_rng"][0]:.5f} {r["a_hyper_rng"][1]:.5f} an {r["a_node_med"]:.5f} {r["a_node_rng"][0]:.5f} {r["a_node_rng"][1]:.5f}')
open(os.path.join(RESULTS, 'junction_transient.txt'), 'w').write('\n'.join(L) + '\n')
json.dump(dict(runs=res, summary=summ), open(os.path.join(RESULTS, 'junction_transient.json'), 'w'), indent=1)
print('\n'.join(L))
