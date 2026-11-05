"""Forecast figure: do 40,000-frame rollouts of the final model (roll_forecast.py) keep the MD clustering and depth
distributions of the held-out second halves? Pair-distance and node-depth densities in the first and last 2,000 rollout
frames against all of the second-half MD and against its last 2,000 frames, and the mean largest cluster in every
2,000-frame window next to MD's own windows. Writes figures/fig_forecast.pdf (and .png) and results/forecast_stats_half.json,
which also holds the plateau (windows 11-20, frames 20,000 to 40,000) per run.
usage: forecast.py   (FC_SYSTEMS restricts a test run)"""
import json
import os

import numpy as np

import figures as F
from common import SYSTEMS, forecast_picks, heldout_lk, reference
from figures import plt
from membrane import fit_objective as fo
from paths import RESULTS, ROLLS40K

H = heldout_lk()

ROLLS = ROLLS40K
STEPS, STRIDE, WIN = 40000, 4, 2000
NW, K = STEPS // WIN, WIN // STRIDE                      # windows, kept frames per window
SHOW = (0, NW - 1)                                       # windows drawn as densities
C_WIN = ('#d4685a', '#a8301f')                           # first and last window: later is darker
C_MDFILL = '#d3d3d3'


def lnp(pd):
    return np.log(np.maximum(pd, 1.0)).ravel()


def window_stats(s):
    box, rc, nm, P = s['box'], s['rc'], s['nm'], s['P']
    rnd = np.random.default_rng(0).random((8000, P, 2)) * box
    lr = H.lnpd(rnd, box).ravel()
    out = {'md': {}, 'seeds': {}, 'rnd_big': float(fo.largest_cluster(fo.pair_dist_xy(np.random.default_rng(0).random((100000, P, 2)) * box, box), rc).mean())}   # as fo.Reference
    for k, v in s['md'].items():
        pd_all = fo.pair_dist_xy(v[::STRIDE], box)
        pd_last = fo.pair_dist_xy(v[-WIN::STRIDE], box)
        lm, ll = lnp(pd_all), lnp(pd_last)
        dep = s['depth'][k]['node'] * nm
        dl = dep[-WIN:]
        bw = [float(fo.largest_cluster(fo.pair_dist_xy(v[a:a + WIN:STRIDE], box), rc).mean()) for a in range(0, len(v) - WIN + 1, WIN)]
        out['md'][k] = dict(lm=lm, ll=ll, kr=H.gparts(lm, lr)[0], kr_last=H.gparts(ll, lr)[0], len=len(v),
                            big=float(fo.largest_cluster(fo.pair_dist_xy(v, box), rc).mean()), big_windows=bw,   # all MD frames, as fo.Reference
                            pd_last_h=np.histogram(pd_last.ravel() * nm, s['edges'])[0],
                            pd_first_h=np.histogram(fo.pair_dist_xy(v[:WIN:STRIDE], box).ravel() * nm, s['edges'])[0],
                            dep_first_h=np.histogram(dep[:WIN][np.isfinite(dep[:WIN])], s['dep_edges'])[0],
                            dep_last_h=np.histogram(dl[np.isfinite(dl)], s['dep_edges'])[0],
                            dep_q=np.percentile(dep[np.isfinite(dep)], [5, 50, 95]),
                            dep_q_last=np.percentile(dl[np.isfinite(dl)], [5, 50, 95]),
                            dep_q_first=np.percentile(dep[:WIN][np.isfinite(dep[:WIN])], [5, 50, 95]))
    for k, m in out['md'].items():
        o = out['md'][next(x for x in s['md'] if x != k)]
        m['floor'] = H.gparts(m['lm'], o['lm'])[0] / m['kr']
        m['floor_last'] = H.gparts(m['ll'], o['ll'])[0] / m['kr_last']
    for r in s['rows']:
        tag = r['tag']
        per = {}
        for k in s['md']:
            z = np.load(os.path.join(ROLLS, f'{tag}_{k}.npz'))
            xy, nd = z['xy'][:, 1:], z['node_depth']          # both at frames 4, 8, ..., 40000
            assert xy.shape[1] == nd.shape[1] == NW * K, (tag, k, xy.shape, nd.shape)
            m = out['md'][k]
            kap, big, dep_q, pd_h, dep_h = [], [], [], {}, {}
            for w in range(NW):
                pd = fo.pair_dist_xy(xy[:, w * K:(w + 1) * K], box)
                dw = nd[:, w * K:(w + 1) * K] * nm
                d = dw[np.isfinite(dw)]
                kap.append(H.gparts(m['lm'], lnp(pd))[0] / m['kr'])
                big.append(float(fo.largest_cluster(pd, rc).mean()))
                dep_q.append(np.percentile(d, [5, 50, 95]))
                if w in SHOW:
                    pd_h[w] = np.histogram(pd.ravel() * nm, s['edges'])[0]
                    dep_h[w] = np.histogram(d, s['dep_edges'])[0]
                if w == NW - 1:
                    kap_last = H.gparts(m['ll'], lnp(pd))[0] / m['kr_last']
            per[k] = dict(kappa=np.array(kap), big=np.array(big), dep_q=np.array(dep_q), pd_h=pd_h, dep_h=dep_h, kappa_last=kap_last)
        out['seeds'][r['seed']] = per
        print(f"{s['name']} {tag}: largest cluster first/last window " +
              ' '.join(f"{k[-5:]} {per[k]['big'][0]:.2f}/{per[k]['big'][-1]:.2f}" for k in s['md']), flush=True)
    return out


def density(counts, edges):
    c = np.asarray(counts, float)
    return c / (c.sum() * np.diff(edges))


def fig_forecast(S, W):
    fig, axes = plt.subplots(1, 4, figsize=(F.W, 2.0))
    s, w = S['7-PIEZO'], W['7-PIEZO']
    ks, seeds = list(s['md']), list(w['seeds'])
    for ax, (key, edges, mdfirst, xlabel) in zip(axes[:2], (('pd_h', s['edges'], 'pd_first_h', 'pair distance (nm)'),
                                                           ('dep_h', s['dep_edges'], 'dep_first_h', 'node depth (nm)'))):
        if key == 'pd_h':
            md_all = np.mean([density(np.histogram(fo.pair_dist_xy(s['md'][k][::STRIDE], s['box']).ravel() * s['nm'], edges)[0], edges) for k in ks], 0)
        else:
            md_all = np.mean([density(np.histogram(s['depth'][k]['node'][np.isfinite(s['depth'][k]['node'])] * s['nm'], edges)[0], edges) for k in ks], 0)
        ax.fill_between(edges[:-1], md_all, step='post', color=C_MDFILL, lw=0)
        ax.step(edges[:-1], np.mean([density(w['md'][k][mdfirst], edges) for k in ks], 0), where='post', color=F.C_MD, lw=0.7)
        for c, wi, ls in zip(C_WIN, SHOW, ((0, (2, 1.5)), '-')):      # first window dashed (2 pt dashes), last solid
            ax.step(edges[:-1], np.mean([density(w['seeds'][sd][k][key][wi], edges) for sd in seeds for k in ks], 0), where='post', color=c, lw=0.9, ls=ls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('density (1/nm)')
        ax.set_title('7-PIEZO', fontsize=8, pad=2, loc='right')
    for ax, (name, s) in zip(axes[2:], S.items() if False else ((n, S[n]) for n in ('7-PIEZO', '14-PIEZO'))):
        w, ks = W[name], list(s['md'])
        seeds = list(w['seeds'])
        t = (np.arange(NW) + 0.5) * WIN / 1000
        for k, ls, mk in zip(ks, ('-', '--'), ('o', 's')):
            if name == '14-PIEZO' and k == ks[0]:
                continue          # largest cluster not read on this trajectory (MD 3.44 vs random 3.46)
            a = np.array([w['seeds'][sd][k]['big'] for sd in seeds])
            ax.fill_between(t, a.min(0), a.max(0), color=F.C_MOD, alpha=0.15, lw=0)
            ax.plot(t, a.mean(0), ls, color=F.C_MOD, lw=0.9)
            bw = w['md'][k]['big_windows']
            ax.plot((np.arange(len(bw)) + 0.5) * WIN / 1000, bw, ls, color=F.C_MD, lw=0.6, marker=mk, ms=2, mfc='white', mew=0.5)
        ax.axhline(w['rnd_big'], ls=':', color=F.C_RND, lw=0.8)
        ax.set_xlim(0, STEPS / 1000)
        kv = ks[1:] if name == '14-PIEZO' else ks
        top = max(max(max(w['md'][k]['big_windows']) for k in kv), max(w['seeds'][sd][k]['big'].max() for sd in seeds for k in kv))
        ax.set_ylim(1, min(s['P'], top + 0.8))
        ax.set_xlabel('frame (10$^3$)')
        ax.set_ylabel('largest cluster')
        ax.set_title(name, fontsize=8, pad=2, loc='right')
    for ax, c in zip(axes, 'abcd'):
        F.style(ax)
        ax.locator_params(axis='x', nbins=2 if c in 'ab' else 3)
        F.letter(ax, c)
    h = [plt.Rectangle((0, 0), 1, 1, color=C_MDFILL, label='MD, all frames'),
         plt.Line2D([], [], color=F.C_MD, lw=0.8, label='MD, first 2,000 frames'),
         plt.Line2D([], [], color=C_WIN[0], lw=0.9, ls=(0, (2, 1.5)), label='full model, first 2,000'),
         plt.Line2D([], [], color=C_WIN[1], lw=0.9, label='full model, last 2,000'),
         plt.Line2D([], [], color=F.C_RND, lw=0.8, ls=':', label='random placement'),
         plt.Line2D([], [], color='0.3', lw=0.9, marker='o', ms=2.5, mfc='white', label='trajectory 1'),
         plt.Line2D([], [], color='0.3', lw=0.9, ls='--', marker='s', ms=2.5, mfc='white', label='trajectory 2')]
    F.finish(fig, h, 3)
    F.save(fig, 'fig_forecast')


def summary(S, W):
    js = {}
    for name, s in S.items():
        w, ks = W[name], list(s['md'])
        seeds = list(w['seeds'])
        js[name] = {'random_largest_cluster': w['rnd_big'], 'seeds': seeds}
        for k in ks:
            m = w['md'][k]
            js[name][k] = {
                'md_len': m['len'], 'md_largest_cluster_all': m['big'], 'md_largest_cluster_windows': m['big_windows'],
                'md_kappa_floor': m['floor'], 'md_kappa_floor_last2000': m['floor_last'],
                'md_node_depth_q5_50_95_nm_all': m['dep_q'].tolist(), 'md_node_depth_q5_50_95_nm_last2000': m['dep_q_last'].tolist(),
                'md_node_depth_q5_50_95_nm_first2000': m['dep_q_first'].tolist(),
                'model_largest_cluster_by_window_seedmean': np.mean([w['seeds'][sd][k]['big'] for sd in seeds], 0).tolist(),
                'model_largest_cluster_first_last_seeds': [[float(w['seeds'][sd][k]['big'][i]) for sd in seeds] for i in (0, -1)],
                'model_kappa_by_window_seedmedian': np.median([w['seeds'][sd][k]['kappa'] for sd in seeds], 0).tolist(),
                'model_kappa_first_last_seeds': [[float(w['seeds'][sd][k]['kappa'][i]) for sd in seeds] for i in (0, -1)],
                'model_kappa_last_vs_md_last2000_seeds': [float(w['seeds'][sd][k]['kappa_last']) for sd in seeds],
                'model_node_depth_q5_50_95_nm_first_last_seedmean': [np.mean([w['seeds'][sd][k]['dep_q'][i] for sd in seeds], 0).tolist() for i in (0, -1)],
            }
    json.dump(js, open(os.path.join(RESULTS, 'forecast_stats_half.json'), 'w'), indent=1, default=float)
    return js


def plateau(w, k, seeds):
    """per run: mean largest cluster and median κ over windows 11-20, and the mean node-depth quantiles"""
    return {'big': [float(w['seeds'][sd][k]['big'][NW // 2:].mean()) for sd in seeds],
            'kappa': [float(np.median(w['seeds'][sd][k]['kappa'][NW // 2:])) for sd in seeds],
            'dep_q': [w['seeds'][sd][k]['dep_q'][NW // 2:].mean(0).tolist() for sd in seeds]}


if __name__ == '__main__':
    S, W = {}, {}
    for name in [n for n in SYSTEMS if n in os.environ.get('FC_SYSTEMS', ','.join(SYSTEMS)).split(',')]:
        s = dict(SYSTEMS[name])
        s['md'], s['depth'], _, s['box'] = reference(name)
        s['rows'] = [r for r in forecast_picks(name) if all(os.path.exists(os.path.join(ROLLS, f"{r['tag']}_{k}.npz")) for k in s['md'])]
        if len(s['rows']) < len(forecast_picks(name)):
            print('forecast rollouts missing for', [r['tag'] for r in forecast_picks(name) if r not in s['rows']], flush=True)
        cfg = json.load(open(os.path.join(os.path.dirname(s['rows'][0]['path']), 'config.json')))
        s['rc'] = float(cfg.get('fit_rc') or fo.RC)
        s['P'] = next(iter(s['md'].values())).shape[1]
        s['name'] = name
        s['edges'] = np.linspace(0, s['box'] / np.sqrt(2) * s['nm'], 61)
        dmd = np.concatenate([s['depth'][k]['node'] for k in s['md']]) * s['nm']
        lo, hi = np.nanquantile(dmd, [0.001, 0.999])
        s['dep_edges'] = np.linspace(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo), 61)
        S[name], W[name] = s, window_stats(s)
    fig_forecast(S, W)
    js = summary(S, W)
    for name in S:
        js[name]['runs'] = [(r['run'], r['seed'], r['epoch'], r['n_epochs']) for r in S[name]['rows']]
        js[name]['seeds'] = list(W[name]['seeds'])
        for k in S[name]['md']:
            js[name][k]['plateau'] = plateau(W[name], k, list(W[name]['seeds']))
    json.dump(js, open(os.path.join(RESULTS, 'forecast_stats_half.json'), 'w'), indent=1, default=float)
    for name in S:
        print(name, 'random largest cluster', round(js[name]['random_largest_cluster'], 2))
        for k in S[name]['md']:
            v = js[name][k]
            print(f"  {k}: largest cluster plateau {np.mean(v['plateau']['big']):.2f}  MD all {v['md_largest_cluster_all']:.2f} "
                  f"MD windows {[round(x, 2) for x in v['md_largest_cluster_windows']]}; kappa plateau {np.mean(v['plateau']['kappa']):.3f} "
                  f"(floor {v['md_kappa_floor']:.3f})")
