"""Pick each run's checkpoint by fit objective without the diffusion term: argmin of objective - diffusion_excess over the
deduped (last line per epoch) rows of ckpt_nd/<run>/fit_objective.jsonl, ties to the earliest epoch (epoch_signed).
fit_objective.py divides cluster_gap by the signed (MD big - random big); where that is negative (14-PIEZO) the clustering
term is recomputed with the absolute divisor, per trajectory from the stored big_<name>, floor included, and the pick
(epoch) is argmin of clustering_excess_abs + depth_excess + overlap_guard.
Reads assets/checkpoints/<run>/{config.json,fit_objective.jsonl} and the first-half MD; writes results/selection_nd.json, which should equal the
committed reproduce/selection_nd.json.
usage: select_epochs.py"""
import glob
import json
import os
import sys

import numpy as np

from paths import CKPT as ND, RESULTS, ROOT, SELECTION, check_md

SEL = os.path.join(RESULTS, 'selection_nd.json')
PARTS = ('clustering_excess', 'diffusion_excess', 'depth_excess', 'overlap_guard')


def rows(run):
    d = {}
    for line in open(f'{ND}/{run}/fit_objective.jsonl'):
        if line.strip():
            r = json.loads(line)
            d[int(r['epoch'])] = r
    return d


def pick(d, run):
    for e, r in d.items():
        s = sum(r.get(k, 0.0) for k in PARTS)
        assert abs(r['objective'] - s) < 1e-9, (run, e, r['objective'], s)
    old = min(d, key=lambda e: (d[e]['objective'], e))
    new = min(d, key=lambda e: (d[e]['objective'] - d[e].get('diffusion_excess', 0.0), e))
    return new, old


_REFS = {}


def reference(cfg):
    """fo.Reference on the run's first-half MD as the training callback built it, plus the signed and absolute floors"""
    key = (tuple(cfg['stability_gt_paths']), tuple(cfg['membrane_data_paths']), cfg.get('fit_rc'), cfg.get('fit_core'))
    if key not in _REFS:
        sys.path.insert(0, ROOT)
        os.chdir(ROOT)
        from membrane import fit_objective as fo
        from membrane.FitObjectiveCallback import load_depth, load_md
        md = load_md(cfg['stability_gt_paths'])
        box = float(cfg['box_size'])
        ref = fo.Reference(md, box, rc=float(cfg.get('fit_rc') or fo.RC), core=float(cfg.get('fit_core') or fo.CORE),
                           depth=load_depth(md, cfg['membrane_data_paths'], box))
        div = {k: ref.big[k] - ref.rand_big for k in md}
        q = [fo._parts(md[j][None], ref, k, z=False) for k in md for j in md if j != k]
        cg = [x['cluster_gap'] for x in q]
        pg = np.mean([x['pair_gap'] for x in q])
        _REFS[key] = (ref, div, 0.5 * (np.mean(cg) + pg), 0.5 * (np.mean(np.abs(cg)) + pg))
    return _REFS[key]


def abs_scores(run, d):
    """{epoch: score} with the absolute cluster divisor, or None where every divisor is positive"""
    ref, div, f_signed, f_abs = reference(json.load(open(f'{ND}/{run}/config.json')))
    if min(div.values()) > 0:
        return None
    r0 = next(iter(d.values()))
    assert abs(r0['floor_clustering'] - f_signed) < 1e-6, (run, r0['floor_clustering'], f_signed)
    out = {}
    for e, r in d.items():
        cg = np.mean([abs(ref.big[k] - r[f'big_{k}']) / abs(div[k]) for k in div])
        assert abs(np.mean([abs(ref.big[k] - r[f'big_{k}']) / div[k] for k in div]) - r['cluster_gap']) < 1e-6, (run, e)
        ex = max(0.0, 0.5 * (cg + r['pair_gap']) - f_abs)
        out[e] = ex + r.get('depth_excess', 0.0) + r.get('overlap_guard', 0.0)
    return out


def select():
    old = json.load(open(SELECTION))
    out = {}
    for run in sorted(os.path.basename(p) for p in glob.glob(f'{ND}/h*_s7*')):
        d = rows(run)
        signed, prev = pick(d, run)
        sc = abs_scores(run, d)
        new = signed if sc is None else min(sc, key=lambda e: (sc[e], e))
        out[run] = dict(epoch=new, epoch_signed=signed, old_epoch=prev, n_epochs=len(d), source=old.get(run, {}).get('source', '?'))
    json.dump(out, open(SEL, 'w'), indent=1)
    print(f'{sum(out[r] != old.get(r) for r in out)} of {len(out)} runs differ from reproduce/selection_nd.json')
    return out


if __name__ == '__main__':
    check_md(first=True)
    select()
