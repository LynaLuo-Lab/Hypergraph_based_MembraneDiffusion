"""Membrane-hypergraph trajectory sampler. Mirrors scripts/diffusion/eval/sample_traj_generic.py
but builds the membrane model (autoregressive k=2..7 membrane module +
xformer decoder) directly, since scripts/sample_old_checkpoints.load_one_checkpoint
constructs via diffusion.model_registry and would silently drop the membrane path.

The membrane state is seeded per rollout by membrane_module.init_head(coords_0)
inside predict_future (membrane/model_multilag.py), so ICs from ANY csv
work (no GT mesh needed at inference). members/ksize load from membrane_data_path
at model construction.

Output: <out>/traj1.csv .. trajN.csv  (t,x1,y1,z1,...; box-wrapped) -- same format
as sample_traj_generic so scripts/shared/analysis/analyze_trajectories.py consumes it unchanged.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = next(d for d in (os.path.abspath(__file__ + '/..' * n) for n in range(1, 6)) if os.path.isfile(d + '/pyproject.toml'))
sys.path.insert(0, ROOT)
from membrane.model_multilag import BrownianModel


def _extract_cfg(ck):
    """Pull the flat config dict out of the checkpoint hyper_parameters."""
    hp = ck.get('hyper_parameters', {})
    if isinstance(hp, dict) and 'config' in hp and isinstance(hp['config'], dict):
        return dict(hp['config'])
    return dict(hp)


def load_membrane_checkpoint(ckpt_path, device, overrides=None):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = _extract_cfg(ck)
    if overrides:
        cfg.update(overrides)          # e.g. squash_enabled for the frozen-backbone trainer
    # Inference: no compile (ROCm one-off), no gradient checkpointing.
    cfg['disable_torch_compile'] = True
    cfg['gradient_checkpointing'] = False
    cfg['use_gradient_checkpointing'] = False
    cfg.setdefault('box_size', 856.196)
    if not cfg.get('membrane_hypergraph'):
        raise SystemExit(f"{ckpt_path} is not a membrane_hypergraph checkpoint "
                         f"(membrane_hypergraph={cfg.get('membrane_hypergraph')})")

    model = BrownianModel(config=cfg, box_size=cfg['box_size'])

    # Strip torch.compile '_orig_mod.' prefix from checkpoint keys, then remap
    # onto the (possibly-compiled) model key names.
    sd_clean = {k.replace('_orig_mod.', ''): v for k, v in ck['state_dict'].items()}
    mkc = {k: k.replace('_orig_mod.', '') for k in model.state_dict().keys()}
    remapped = {mk: sd_clean[ck_k] for mk, ck_k in mkc.items() if ck_k in sd_clean}
    result = model.load_state_dict(remapped, strict=False)
    # membrane rollout MUST have every weight; surface any gap loudly.
    miss = [k for k in result.missing_keys if 'membrane' in k or 'decoder' in k
            or 'encoder' in k or 'geo_embed' in k or 'stochastic' in k or 'transition' in k]
    if miss:
        raise SystemExit(f"  FATAL missing weights ({len(result.missing_keys)}): {miss[:8]}")
    if result.missing_keys:
        print(f"  WARN missing ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  WARN unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")

    pm = sum(p.numel() for p in model.parameters())
    pc = sum(v.numel() for v in ck['state_dict'].values())
    print(f"  params model={pm:,}  ckpt={pc:,}  delta={pc-pm:+,}  loaded={len(remapped):,}", flush=True)
    model.to(device).eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gt', required=True, help='csv for IC pool (t, then n_particles*3 coords)')
    ap.add_argument('--n_particles', type=int, default=7)
    ap.add_argument('--n_traj', type=int, default=8)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--ic_seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    nc = args.n_particles * 3

    gt = np.loadtxt(args.gt, delimiter=',')
    pool = torch.tensor(gt[:, 1:1 + nc], dtype=torch.float32)
    g = torch.Generator().manual_seed(args.ic_seed)
    ic = pool[torch.randperm(pool.shape[0], generator=g)[:args.n_traj]].contiguous()

    model, cfg = load_membrane_checkpoint(args.ckpt, dev)
    box = float(cfg.get('box_size', 856.196))
    print(f"  loaded {args.ckpt}  box={box:.4f}  n_particles={args.n_particles}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    cols = [f'{a}{p}' for p in range(1, args.n_particles + 1) for a in 'xyz']
    tcol = np.arange(args.steps, dtype=np.int64)

    with torch.no_grad():
        k = 0
        i = 0
        while i < ic.shape[0]:
            xb = torch.remainder(ic[i:i + args.chunk].to(dev), box).unsqueeze(1)  # [b,1,nc]
            sw = model.predict_future(xb, steps=args.steps)                        # [b,steps,nc]
            sw = sw.reshape(xb.shape[0], args.steps, nc).cpu().numpy()
            for pos in sw:
                df = pd.DataFrame(pos, columns=cols)
                df.insert(0, 't', tcol)
                df.to_csv(os.path.join(args.out, f'traj{k+1}.csv'), index=False, float_format='%.5f')
                k += 1
            i += args.chunk
            print(f"  rolled {k}/{ic.shape[0]} traj", flush=True)
    print(f"DONE {args.out}: {k} trajectories x {args.steps} steps")


if __name__ == '__main__':
    main()
