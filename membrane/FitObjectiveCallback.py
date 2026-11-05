"""Scores membrane/fit_objective.py at the end of every epoch: N free rollouts from frame 0 of each MD trajectory,
as long as that trajectory. Logs fit/objective and its parts, prints one line, and appends one JSON line per epoch
to <dirpath>/fit_objective.jsonl so a weight search can read the whole curve without any checkpoint.

A hypergraph membrane model is scored on depth too, against the depth in the membrane npz files that cover the MD
frames; its depth comes out divided by the std the membrane pool normalises by and is multiplied back."""
import json
import os

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from data.paths import resolve
from membrane import fit_objective as fo

LOGGED = ('objective', 'clustering', 'cluster_gap', 'pair_gap', 'diffusion', 'z_out', 'depth', 'node_gap', 'hyper_gap',
          'coupling_gap', 'overlap', 'clustering_excess', 'diffusion_excess', 'depth_excess',
          'overlap_guard', 'drift')


def load_md(paths):
    """rows: timestep, then x y z per protein, wrapped model units -> {file stem: [T, P, 3]}"""
    out = {}
    for p in paths:
        a = np.loadtxt(resolve(p), delimiter=',')[:, 1:]
        out[os.path.splitext(os.path.basename(p))[0]] = a.reshape(len(a), -1, 3)
    return out


def start_thetas(md, npz_paths, box):
    """arm angles of the frame each rollout starts from: the npz whose first frame is that MD frame"""
    out = {}
    for k, v in md.items():
        for p in npz_paths or ():
            d = np.load(resolve(p), allow_pickle=True)
            if 'theta' not in d:
                continue
            e = np.abs(d['coords'][0].reshape(-1) - v[0].reshape(-1)); e = np.minimum(e, box - e)
            if e.max() < 1e-2:
                out[k] = d['theta'][0].astype(np.float32)
    return out


def load_depth(md, npz_paths, box):
    """{name: {'node': [T, P], 'hyper': [T, H]}} for the MD trajectories some npz covers: each npz put at the frames
    of the trajectory its coordinates match, NaN on frames no npz has"""
    out = {}
    for p in npz_paths or ():
        d = np.load(resolve(p), allow_pickle=True)
        if 'node_depth' not in d:
            continue
        c = d['coords'].reshape(len(d['coords']), -1).astype(np.float64)
        for k, v in md.items():
            e = np.abs(v.reshape(len(v), -1) - c[0]); e = np.minimum(e, box - e).max(-1)
            s = int(e.argmin())
            if e[s] >= 1e-2 or s + len(c) > len(v):
                continue
            e = np.abs(v[s:s + len(c)].reshape(len(c), -1) - c); e = np.minimum(e, box - e)
            if e.max() >= 1e-2:
                raise ValueError(f'{p} starts at frame {s} of {k} and then leaves it')
            o = out.setdefault(k, {'node': np.full((len(v), d['node_depth'].shape[1]), np.nan),
                                   'hyper': np.full((len(v), d['hyper_depth'].shape[1]), np.nan)})
            o['node'][s:s + len(c)], o['hyper'][s:s + len(c)] = d['node_depth'], d['hyper_depth']
    return out


def depth_std(npz_paths):
    """(node, hyper) std the membrane pool divides depth by: one std over all files (data/membrane_dataset.py)"""
    ds = [np.load(resolve(p), allow_pickle=True) for p in npz_paths]
    return tuple(float(np.concatenate([d[f].astype(np.float32).ravel() for d in ds]).std()) or 1.0
                 for f in ('node_depth', 'hyper_depth'))


@torch.inference_mode()
def fit_rollouts(model, md, thetas, n, seed=2000, depth_scale=None):
    """{name: wrapped [n, T, P, 3]} with the MD start frame first. Leaves the global RNG where it found it.
    With depth_scale, the (node, hyper) std the model's depth is divided by, also {name: (node [n, T-1, P],
    hyper [n, T-1, H])}, the depth at frames 1.. in npz units."""
    cpu, gpu = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    box, dev, out, dep = float(model.box_size), model.device, {}, {}
    try:
        for k, v in md.items():
            x0 = torch.remainder(torch.tensor(v[0].reshape(1, 1, -1), dtype=torch.float32, device=dev), box).repeat(n, 1, 1)
            kw = {}
            if getattr(model, 'arms_enabled', False) and k in thetas:
                kw['initial_theta'] = torch.tensor(thetas[k], device=dev).unsqueeze(0).repeat(n, 1)
            torch.manual_seed(seed)
            if gpu is not None:
                torch.cuda.manual_seed_all(seed)
            if depth_scale is None:
                sw = model.predict_future(x0, steps=len(v) - 1, return_unwrapped=False, **kw)
            else:
                sw, mn, mh = model.predict_future(x0, steps=len(v) - 1, return_unwrapped=False, return_membrane=True, **kw)
                dep[k] = (mn.float().cpu().numpy() * depth_scale[0], mh.float().cpu().numpy() * depth_scale[1])
            out[k] = torch.cat([x0.float(), sw.float()], 1).reshape(n, len(v), -1, 3).cpu().numpy()
    finally:
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)
    return out if depth_scale is None else (out, dep)


def has_depth(model):
    return bool(getattr(model, 'membrane_mode', False) and getattr(model, 'membrane_hypergraph', False))


class FitObjectiveCallback(Callback):
    def __init__(self, gt_paths, npz_paths=None, n_rollouts=32, dirpath=None, check_interval=1, rc=None, core=None):
        self.gt_paths, self.npz_paths = list(gt_paths), list(npz_paths or [])
        self.n, self.dirpath, self.check_interval = int(n_rollouts), dirpath, int(check_interval)
        # cluster cutoff and hard core in model units; the defaults are 7-PIEZO's
        self.rc = fo.RC if rc is None else float(rc)
        self.core = fo.CORE if core is None else float(core)
        self.ref = None

    def _ensure(self, pl_module):
        if self.ref is None:
            box = float(pl_module.box_size)
            self.md = load_md(self.gt_paths)
            n = getattr(pl_module, 'n_particles', None)
            widths = sorted({v.shape[1] for v in self.md.values()})
            if n is not None and widths != [int(n)]:
                raise ValueError(f'MD reference trajectories {self.gt_paths} hold {widths} proteins, model n_particles={n}')
            self.scale, dep = None, None
            if has_depth(pl_module):
                dep = load_depth(self.md, self.npz_paths, box)
                if set(dep) != set(self.md):
                    raise RuntimeError(f"no membrane npz has depth for {sorted(set(self.md) - set(dep))}")
                self.scale = depth_std(self.npz_paths)
            self.ref = fo.Reference(self.md, box, rc=self.rc, core=self.core, depth=dep)
            self.thetas = start_thetas(self.md, self.npz_paths, box)
            if getattr(pl_module, 'arms_enabled', False) and set(self.thetas) != set(self.md):
                raise RuntimeError(f"no arms npz starts at frame 0 of {sorted(set(self.md) - set(self.thetas))}")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.check_interval != 0:
            return
        self._ensure(pl_module)
        torch.cuda.empty_cache()
        pl_module.eval()
        rolls = fit_rollouts(pl_module, self.md, self.thetas, self.n, depth_scale=self.scale)
        pl_module.train()
        rolls, dep = rolls if self.scale is not None else (rolls, None)
        s = fo.score(rolls, self.ref, depth=dep, z=getattr(pl_module, 'predict_z', True))
        pl_module.log_dict({f'fit/{k}': float(s[k]) for k in LOGGED if k in s}, prog_bar=False, logger=True)
        print(f"fit epoch {trainer.current_epoch}: objective {s['objective']:.4f}  "
              + " ".join(f"{k} {s[k]:.3f}" for k in LOGGED[1:] if k in s), flush=True)
        if self.dirpath and trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
            with open(os.path.join(self.dirpath, 'fit_objective.jsonl'), 'a') as f:
                f.write(json.dumps({'epoch': int(trainer.current_epoch), **s}) + '\n')
