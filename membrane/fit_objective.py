"""Objective for fitting the membrane loss weights: clustering, lateral diffusion and membrane depth of free
rollouts against MD. Rollouts start at frame 0 of each MD trajectory and run as long as it; every part is 0 for a
model that reproduces MD.

    clustering     0 = MD, 1 = seven independent uniform points in the box. Mean of two gaps, each averaged over
                   the MD trajectories: size of the largest cluster (proteins linked when closer than rc in x,y),
                   and the distance between the pooled pair-distance distributions.
    diffusion      mean over lags and trajectories of |ln(model MSD / MD MSD)|, x,y.
    depth          membrane depth, when the reference has MD depth: mean of three gaps, each averaged over the MD
                   trajectories. Protein and hyperedge depth, the distance between the pooled distributions over
                   MD's standard deviation; and coupling, |r(model) - r(MD)| for r the correlation of a protein's
                   depth with its mean x,y distance to the others (MD 0.9 and over: isolated proteins sit deeper).
    overlap        share of pairs closer than the hard core.

Clustering, diffusion and depth are charged only above their MD floor, the same part with one MD trajectory
scored against the other: two MD runs differ by that much, so a model closer than that is not told apart.
overlap_guard is charged only above OVERLAP_FREE, the range where a squash head trained afterwards was measured to
remove the overlap (2.3-2.9 % -> 0.12-0.24 %) at a clustering cost of at most 0.04 (results/membrane/squash-head/2026-10-15_squash-head-check/squashcheck), so
tighter clusters cannot be bought with proteins sitting on top of each other beyond what the head can undo.
objective is the sum of the excesses and the guard.

Reported next to it, not part of the sum:

    z_out          share of z positions outside MD's 0.5-99.5 % band. Not charged: a protein's height is the depth under
                   it plus a constant (slope 0.99, r 0.996 in MD), so it is the depth scored again. Left out, with
                   zcoupling, for a model without z (score(..., z=False)).
    drift          size of the mean x,y displacement over DRIFT_LAG frames, averaged over rollouts, start
                   frames and proteins. Rollouts share a start and differ in noise, so this is the bias a
                   match on displacement sizes cannot see.
    zcoupling      the depth coupling read off the proteins' own z instead of the depth module: r of a protein's
                   height, relative to the others in its frame, with its isolation. MD has it as strongly as
                   depth; a model can get depth right with z ignoring it.

All lengths in model units, depth in the npz's (delta_std, the same unit).
"""
import numpy as np

RC = 68.93 / 0.4018202            # nm -> model units; the 7-PIEZO draft's cluster cutoff
CORE = 53.2                       # hard-core distance, model units
LAGS = (1, 10, 100, 1000, 3000)
OVERLAP_FREE = 0.025
FLOORED = ('clustering', 'diffusion', 'depth')
DEPTH_GAPS = ('node_gap', 'hyper_gap', 'coupling_gap')
DIVERGED = 10.0
DRIFT_LAG = 1000
NQ = 2000


def pair_dist_xy(pos, box):
    """[..., P, >=2] -> minimum-image x,y distance of every pair, [..., P(P-1)/2]"""
    i, j = np.triu_indices(pos.shape[-2], 1)
    d = pos[..., i, :2] - pos[..., j, :2]
    d = d - box * np.round(d / box)
    return np.sqrt((d ** 2).sum(-1))


def isolation(pos, box):
    """[..., P, >=2] -> each protein's mean minimum-image x,y distance to the other P-1, [..., P]"""
    P = pos.shape[-2]
    pd = pair_dist_xy(pos, box)
    out = np.zeros(pd.shape[:-1] + (P,))
    for n, (i, j) in enumerate(zip(*np.triu_indices(P, 1))):
        out[..., i] += pd[..., n]; out[..., j] += pd[..., n]
    return out / (P - 1)


def coupling(pos, node, box):
    """correlation of a protein's depth with its isolation, pooled over rollouts, frames and proteins"""
    return float(np.corrcoef(isolation(pos, box).ravel(), np.asarray(node).ravel())[0, 1])


def zcoupling(pos, box):
    """correlation of a protein's height with its isolation, both relative to the other proteins in its frame,
    pooled: which protein sits higher, free of where the slab is. Min-image heights, so a slab across the z wrap stays
    in one piece"""
    h = pos[..., 2] - pos[..., :1, 2]
    h = h - box * np.round(h / box)
    iso = isolation(pos, box)
    return float(np.corrcoef((iso - iso.mean(-1, keepdims=True)).ravel(), (h - h.mean(-1, keepdims=True)).ravel())[0, 1])


def largest_cluster(pd, rc):
    """size of the largest connected group per frame, [...]"""
    P = int(round((1 + np.sqrt(1 + 8 * pd.shape[-1])) / 2))
    i, j = np.triu_indices(P, 1)
    reach = np.zeros(pd.shape[:-1] + (P, P), dtype=bool)
    reach[..., i, j] = reach[..., j, i] = pd <= rc
    reach |= np.eye(P, dtype=bool)
    for _ in range(int(np.ceil(np.log2(P)))):               # path length doubles each pass
        r = reach.astype(np.uint8)
        reach = np.einsum('...ij,...jk->...ik', r, r) > 0
    return reach.sum(-1).max(-1)


def msd_xy(pos, box, lags):
    """wrapped [..., T, P, >=2] -> x,y mean squared displacement per lag, averaged over time and proteins"""
    d = np.diff(pos[..., :2], axis=-3)
    d = d - box * np.round(d / box)
    u = np.concatenate([np.zeros_like(d[..., :1, :, :]), np.cumsum(d, axis=-3)], axis=-3)
    return np.stack([((u[..., L:, :, :] - u[..., :-L, :, :]) ** 2).sum(-1).mean((-1, -2)) for L in lags], -1)


def drift_xy(pos, box, lag):
    """wrapped [N, T, P, >=2] -> size of the mean x,y displacement over lag"""
    d = np.diff(pos[..., :2], axis=-3)
    d = d - box * np.round(d / box)
    u = np.cumsum(d, axis=-3)
    return float(np.linalg.norm((u[..., lag:, :, :] - u[..., :-lag, :, :]).mean((0, 1, 2))))


def _q(a):
    return np.quantile(np.asarray(a).ravel(), (np.arange(NQ) + 0.5) / NQ)


def w1(a, b):
    return float(np.abs(_q(a) - _q(b)).mean())


def trial_value(objs):
    """one number for a training run from its per-epoch objectives: the median of the second half"""
    objs = list(objs)
    return float(np.median(objs[len(objs) // 2:])) if objs else DIVERGED


def last_depth(rows):
    """a run's stored epochs with every depth part set to the last epoch's, before they are charged for a trial value:
    depth still comes down over the epochs the value is taken from, and a run trained before depth was scored has
    it only from its kept last checkpoint, so both are valued on the depth they end with. A last epoch without depth
    (diverged) leaves the rows as stored."""
    if not rows or 'depth' not in rows[-1]:
        return list(rows)
    return [dict(r, depth=rows[-1]['depth']) if 'depth' in r else r for r in rows]


def seed_value(values, lam=1.0):
    """one number for a recipe trained on several seeds: mean + lam * sd, the score one run can be counted on for"""
    v = np.asarray(list(values), dtype=float)
    return float(v.mean() + (lam * v.std(ddof=1) if len(v) > 1 else 0.0))


def depth_frames(d):
    """frames of an MD trajectory that have depth, without frame 0 (a rollout predicts none for its start frame)"""
    ok = np.isfinite(d['node']).all(-1) & np.isfinite(d['hyper']).all(-1)
    ok[0] = False
    return np.flatnonzero(ok)


class Reference:
    def __init__(self, md, box, rc=RC, core=CORE, lags=LAGS, n_random=100000, seed=0, depth=None):
        """md: {name: wrapped [T, P, 3]}; depth: {name: {'node': [T, P], 'hyper': [T, H]}} for every name, NaN on the
        frames MD has no depth for"""
        self.box, self.rc, self.core, self.lags = float(box), rc, core, tuple(lags)
        self.depth = depth is not None
        if self.depth and set(depth) != set(md):
            raise ValueError(f'MD depth for {sorted(depth)}, trajectories {sorted(md)}')
        self.nodeq, self.hyperq, self.node_sd, self.hyper_sd, self.r, dmd = {}, {}, {}, {}, {}, {}
        for k, d in (depth or {}).items():
            f = depth_frames(d)
            node, hyper = np.asarray(d['node'], np.float64)[f], np.asarray(d['hyper'], np.float64)[f]
            self.nodeq[k], self.node_sd[k] = _q(node), float(node.std())
            self.hyperq[k], self.hyper_sd[k] = _q(hyper), float(hyper.std())
            self.r[k] = coupling(md[k][f], node, box)
            dmd[k] = (md[k][f][None], node[None], hyper[None])
        P = next(iter(md.values())).shape[-2]
        rnd = pair_dist_xy(np.random.default_rng(seed).random((n_random, P, 2)) * box, box)
        self.rand_big, rq = float(largest_cluster(rnd, rc).mean()), _q(rnd)
        self.big, self.pdq, self.rand_w1, self.msd, self.frames = {}, {}, {}, {}, {}
        sh = lambda z: np.remainder(z + box / 2, box)
        self.zband = tuple(np.percentile(np.concatenate([sh(v[..., 2]).ravel() for v in md.values()]), (0.5, 99.5)))
        for k, v in md.items():
            pd = pair_dist_xy(v, box)
            self.big[k], self.pdq[k] = float(largest_cluster(pd, rc).mean()), _q(pd)
            self.rand_w1[k] = float(np.abs(rq - self.pdq[k]).mean())
            self.msd[k], self.frames[k] = msd_xy(v, box, self.lags), len(v)
        # each MD trajectory scored as a rollout against every other one, averaged like the model's parts
        pairs = [(k, j) for k in md for j in md if j != k]
        self.floor = {p: 0.0 for p in FLOORED if p != 'depth' or self.depth}
        if pairs:
            f = {k: [] for k in ('cluster_gap', 'pair_gap', 'diffusion') + DEPTH_GAPS}
            for k, j in pairs:
                q = _parts(md[j][None], self, k, z=False)
                if self.depth:
                    q.update(_depth_parts(*dmd[j], self, k))
                for p, v in q.items():
                    if p in f:
                        f[p].append(v)
            m = {p: float(np.mean(v)) for p, v in f.items() if v}
            self.floor = {'clustering': 0.5 * (m['cluster_gap'] + m['pair_gap']), 'diffusion': m['diffusion']}
            if self.depth:
                self.floor['depth'] = float(np.mean([m[p] for p in DEPTH_GAPS]))


def _parts(x, ref, k, z=True):
    """one trajectory's rollouts [N, T, P, 3] against MD trajectory k"""
    x = np.asarray(x, dtype=np.float64)
    pd = pair_dist_xy(x, ref.box)
    big = float(largest_cluster(pd, ref.rc).mean())
    m = msd_xy(x, ref.box, ref.lags).mean(0)
    out = {'big': big, 'msd': m,
           'cluster_gap': abs(ref.big[k] - big) / (ref.big[k] - ref.rand_big),
           'pair_gap': float(np.abs(_q(pd) - ref.pdq[k]).mean()) / ref.rand_w1[k],
           'diffusion': float(np.abs(np.log(np.maximum(m, 1e-12) / ref.msd[k])).mean()),
           'overlap': float((pd < ref.core).mean())}
    if z:
        h = np.remainder(x[..., 2] + ref.box / 2, ref.box)
        out['z_out'] = float(((h < ref.zband[0]) | (h > ref.zband[1])).mean())
    return out


def _depth_parts(x, node, hyper, ref, k):
    """depth [N, F, P] and [N, F, H] with the positions they go with, [N, F, P, 3], against MD trajectory k"""
    node, hyper = np.asarray(node, np.float64), np.asarray(hyper, np.float64)
    r = coupling(x, node, ref.box)
    return {'node_gap': float(np.abs(_q(node) - ref.nodeq[k]).mean()) / ref.node_sd[k],
            'hyper_gap': float(np.abs(_q(hyper) - ref.hyperq[k]).mean()) / ref.hyper_sd[k],
            'coupling_gap': abs(r - ref.r[k]), 'r': r,
            'node_sd_ratio': float(node.std()) / ref.node_sd[k], 'hyper_sd_ratio': float(hyper.std()) / ref.hyper_sd[k]}


def charge(parts, floor):
    """the excess of each part over its MD floor, the overlap guard and their sum, from the averaged parts (also
    those stored in fit_objective.jsonl, so a finished run can be rescored). A row scored without depth is charged
    without it."""
    out = {f'{p}_excess': max(0.0, parts[p] - floor[p]) for p in FLOORED if p in parts}
    out['overlap_guard'] = max(0.0, parts['overlap'] - OVERLAP_FREE) / OVERLAP_FREE
    out['objective'] = sum(out.values())
    return out


def score(rolls, ref, depth=None, z=True):
    """rolls: {name: wrapped [N, T, P, 3]}, same names as the reference. depth, required when the reference has MD
    depth: {name: (node [N, T-1, P], hyper [N, T-1, H])}, the depth at frames 1.. of the rollouts. z=False: the model
    has no protein height, so z_out and zcoupling are left out."""
    if ref.depth and depth is None:
        raise ValueError('the reference has MD depth and these rollouts none: pass depth')
    part = {k: [] for k in ('cluster_gap', 'pair_gap', 'diffusion') + (('z_out',) if z else ()) + ('overlap',)
            + (DEPTH_GAPS if ref.depth else ())}
    out = {}
    for k, x in rolls.items():
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all() or (ref.depth and not all(np.isfinite(a).all() for a in depth[k])):
            return {'objective': DIVERGED, 'diverged': 1.0}
        q = _parts(x, ref, k, z=z)
        if ref.depth:
            if any(np.shape(a)[:2] != x[:, 1:].shape[:2] for a in depth[k]):
                raise ValueError(f'{k}: depth {[np.shape(a) for a in depth[k]]} does not cover frames 1.. of {x.shape}')
            q.update(_depth_parts(x[:, 1:], *depth[k], ref, k))
            out.update({f'{p}_{k}': q[p] for p in ('node_sd_ratio', 'hyper_sd_ratio')}, **{f'coupling_{k}': q['r']})
        for p in part:
            part[p].append(q[p])
        out[f'big_{k}'] = q['big']
        if z:
            out[f'zcoupling_{k}'] = zcoupling(x, ref.box)
        if x.shape[1] > DRIFT_LAG:
            out[f'drift_{k}'] = drift_xy(x, ref.box, DRIFT_LAG)
        for L, r in zip(ref.lags, q['msd'] / ref.msd[k]):
            out[f'msd_ratio_{k}_{L}'] = float(r)
    out.update({k: float(np.mean(v)) for k, v in part.items()})
    dr = [v for k, v in out.items() if k.startswith('drift_')]
    if dr:
        out['drift'] = float(np.mean(dr))
    out['clustering'] = 0.5 * (out['cluster_gap'] + out['pair_gap'])
    if ref.depth:
        out['depth'] = float(np.mean([out[p] for p in DEPTH_GAPS]))
    out.update({f'floor_{p}': ref.floor[p] for p in ref.floor})
    out.update(charge(out, ref.floor))
    return out
