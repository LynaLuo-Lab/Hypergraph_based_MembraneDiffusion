"""Figures and summary numbers for the held-out second-half evaluation. Reads
the held-out rollouts of roll_heldout.py (32 per second-half MD trajectory from its frame 0, seed 3000, one checkpoint per
training run) and the second-half MD; model curves pool every rollout of every run. 'model' is the final recipe with its
squash head, 'without squash' the same checkpoints with the head off, 'baseline' the baseline recipe. Writes
figures/*.pdf (and .png) and results/stats_half.json.
usage: figures.py [FIGURE ...]"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from common import NAMES, SEED, SYSTEMS, heldout_lk, picks, reference
from membrane import fit_objective as fo
from paths import CACHE, FIGURES, RESULTS, ROLLS, TRACES

H = heldout_lk()
OUT = FIGURES
STRIDE = 4
LAGS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
SHORT_EDGES = np.linspace(0, 70, 71)     # nm, the range around the hard core (21.4 nm)
C_MD, C_MOD, C_RND, C_BASE, C_OFF, C_LK = 'black', '#c0392b', '#8c8c8c', '#e8a33d', '#3a78b5', '#4c9a5a'
W = 11.95 / 2.54         # figure width before the tight crop, which lands at the 12.2 cm text width, inches
VARIANTS = ('sq', 'base', 'nosq', 'lk')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.labelpad': 2, 'savefig.pad_inches': 0.02})


G_ROWS = {}


def setup(name):
    """second-half MD, depth, reference and the rows of every pick; rows are joined to picks on the tag and checked on (run, seed)"""
    s = dict(SYSTEMS[name])
    s['md'], s['depth'], _, s['box'] = reference(name)
    G_ROWS.update({r['tag']: r for r in map(json.loads, open(os.path.join(RESULTS, 'g_half.jsonl')))})
    rows = {}
    for r in map(json.loads, open(os.path.join(RESULTS, 'rows_half.jsonl'))):
        rows[r['tag']] = r
    s['picks'], s['rows'] = {}, {}
    for v in VARIANTS:
        s['picks'][v] = sorted(picks(name, (v,)), key=lambda p: p['seed'])
        s['rows'][v] = []
        for p in s['picks'][v]:
            r = rows[p['tag']]
            assert (r['run'], r['seed'], r['epoch'], r['variant']) == (p['run'], p['seed'], p['epoch'], v), (p['tag'], r['tag'])
            r['g'], r['g_floor'] = G_ROWS[p['tag']]['g'], G_ROWS[p['tag']]['g_floor']
            s['rows'][v].append(r)
    assert [r['seed'] for r in s['rows']['sq']] == [r['seed'] for r in s['rows']['nosq']], 'sq / nosq seeds'
    cfg = json.load(open(os.path.join(os.path.dirname(s['picks']['sq'][0]['path']), 'config.json')))
    s['rc'], s['core'] = float(cfg.get('fit_rc') or fo.RC), float(cfg.get('fit_core') or fo.CORE)
    s['P'] = next(iter(s['md'].values())).shape[1]
    s['ref'] = fo.Reference(s['md'], s['box'], rc=s['rc'], core=s['core'], depth=s['depth'])
    return s


def survival(c):
    """c [T, pairs] contact booleans -> P(contact at t+L | contact at t) per lag in LAGS (NaN past the length)"""
    out = np.full(len(LAGS), np.nan)
    for i, L in enumerate(LAGS):
        if L < len(c):
            a = c[:-L]
            n = a.sum()
            out[i] = (a & c[L:]).sum() / n if n else np.nan
    return out


def stats(xy, depth, s, edges, iso_edges, dep_edges):
    """one trajectory [T, P, 2] (and node depth [T', P] matched to frames 1.. or None) -> per-trajectory statistics"""
    pd = fo.pair_dist_xy(xy[::STRIDE], s['box'])
    big = fo.largest_cluster(pd, s['rc'])
    out = {'pd': np.histogram(pd.ravel() * s['nm'], edges, density=True)[0],
           'pd_short': np.histogram(pd.ravel() * s['nm'], SHORT_EDGES)[0] / (pd.size * np.diff(SHORT_EDGES)),
           'big': np.bincount(big, minlength=s['P'] + 1)[1:] / len(big),
           'surv': survival(fo.pair_dist_xy(xy, s['box']) <= s['rc'])}
    if depth is not None:
        iso = fo.isolation(xy, s['box']) * s['nm']
        d = depth * s['nm']
        ok = np.isfinite(d).all(-1)
        iso, d = iso[ok].ravel(), d[ok].ravel()
        idx = np.digitize(iso, iso_edges) - 1
        out['dep_iso'] = np.array([np.median(d[idx == b]) if (idx == b).sum() > 20 else np.nan for b in range(len(iso_edges) - 1)])
        out['dep_hist'] = np.histogram(d, dep_edges, density=True)[0]
        out['r'] = float(np.corrcoef(iso, d)[0, 1])
        out['dep_median'] = float(np.median(d))
    return out


def collect(name):
    s = setup(name)
    lim = s['box'] / np.sqrt(2) * s['nm']
    s['edges'] = np.linspace(0, lim, 61)
    allmd = np.concatenate([v[..., :2] for v in s['md'].values()])
    iso_md = fo.isolation(allmd, s['box']) * s['nm']
    s['iso_edges'] = np.quantile(iso_md, np.linspace(0.01, 0.99, 15))
    dmd = np.concatenate([s['depth'][k]['node'] for k in s['md']]) * s['nm']
    lo, hi = np.nanquantile(dmd, [0.001, 0.999])
    s['dep_edges'] = np.linspace(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo), 61)
    s['md_stats'] = {k: stats(v[..., :2], s['depth'][k]['node'], s, s['edges'], s['iso_edges'], s['dep_edges'])
                     for k, v in s['md'].items()}
    rnd = np.random.default_rng(0).random((100000, s['P'], 2)) * s['box']     # as many as the selection objective's reference
    rpd = fo.pair_dist_xy(rnd, s['box'])
    s['rnd_pd'] = np.histogram(rpd.ravel() * s['nm'], s['edges'], density=True)[0]
    rb = fo.largest_cluster(rpd, s['rc'])
    s['rnd_big'] = np.bincount(rb, minlength=s['P'] + 1)[1:] / len(rb)
    os.makedirs(CACHE, exist_ok=True)
    s['roll_stats'] = {}
    for v in VARIANTS:
        s['roll_stats'][v] = {k: [] for k in s['md']}
        for p in s['picks'][v]:
            cf = os.path.join(CACHE, f"{p['tag']}.pkl")
            if os.path.exists(cf):
                per = pickle.load(open(cf, 'rb'))
            else:
                z = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
                per = {k: [stats(z[f'xy_{k}'][i, 1:], z[f'node_depth_{k}'][i], s, s['edges'], s['iso_edges'], s['dep_edges'])
                           for i in range(len(z[f'xy_{k}']))] for k in s['md']}
                pickle.dump(per, open(cf, 'wb'))
            for k in s['md']:
                s['roll_stats'][v][k] += per[k]
            print(f"{name}: {p['tag']} done", flush=True)
    return s


def pooled_md(s, key, ks=None):
    """the two MD trajectories weighted equally, as the model bands weight their rollouts from each start"""
    return np.nanmean([s['md_stats'][k][key] for k in (ks or s['md'])], 0)


def band(s, key, variant='sq', ks=None):
    a = np.array([r[key] for k in (ks or s['md']) for r in s['roll_stats'][variant][k]])
    return np.nanmedian(a, 0), np.nanpercentile(a, 5, 0), np.nanpercentile(a, 95, 0)


def style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    for axis, sc in (('x', ax.get_xscale()), ('y', ax.get_yscale())):
        if sc == 'linear' and not (axis == 'x' and getattr(ax, '_fixed_x', False)):
            ax.locator_params(axis=axis, nbins=3 if axis == 'x' else 4)


def letter(ax, c):
    ax.text(-0.2, 1.02, c, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def save(fig, name, tight=True):
    """tight bbox with a small pad, legend inside it; LaTeX includes the PDF at scale 1"""
    os.makedirs(OUT, exist_ok=True)
    kw = {}
    if tight:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = [fig.get_tightbbox(r).extents] + [x.get_tightbbox(r).extents / fig.dpi for a in fig.axes for x in (a.xaxis, a.yaxis) if x.get_tightbbox(r) is not None]   # axes.get_tightbbox misses the last x label
        ext = np.array([min(e[0] for e in bb), min(e[1] for e in bb), max(e[2] for e in bb), max(e[3] for e in bb)]) + np.array([-1, -1, 1, 1]) * 0.02
        kw = {'bbox_inches': Bbox.from_extents(*ext)}
    fig.savefig(os.path.join(OUT, name + '.pdf'), **kw)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200, **kw)
    print(name, 'figure size (cm)', round(fig.get_figwidth() * 2.54, 2), round(fig.get_figheight() * 2.54, 2), 'tight' if tight else 'exact', flush=True)
    plt.close(fig)


MODELS = (('base', C_BASE, 'o', False, 'baseline'), ('lk', C_LK, 's', False, 'backbone model'),
          ('nosq', C_OFF, '^', False, 'ablation'), ('sq', C_MOD, 'o', True, 'full model'))


def dots(ax, S, slots, value, floor):
    """per-run values of every model at each slot (system, trajectory) with the run mean as a bar and the floor as a diamond"""
    for i, (n, k) in enumerate(slots):
        for m, (v, c, mk, fill, _) in enumerate(MODELS):
            y = np.array([value(r, k) for r in S[n]['rows'][v]])
            off = -0.33 + 0.22 * m
            ax.plot(i + off + np.linspace(-0.04, 0.04, len(y)), y, mk, ms=2.3, mfc=c if fill else 'none', mec=c, mew=0.7, alpha=0.85 if fill else 1)
            ax.plot([i + off - 0.08, i + off + 0.08], [y.mean()] * 2, color=c, lw=1.2)
        ax.plot(i + 0.47, floor(n, k), 'D', ms=3, color=C_MD)
    ax.axhline(1.0, color=C_RND, ls='--', lw=0.8)
    ax.set_yscale('log')
    ax.set_xlim(-0.55, len(slots) - 0.35)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels([('1', '2')[i % 2] for i in range(len(slots))])
    ax.set_xlabel('trajectory')
    ax._fixed_x = True
    for j, name in enumerate(dict.fromkeys(n for n, _ in slots)):
        ax.annotate(name.replace('PIEZO', 'P'), (2 * j + 0.5, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -21), textcoords='offset points', ha='center', va='top', fontsize=8)
    style(ax)


def panel_pd(ax, s):
    x = 0.5 * (s['edges'][1:] + s['edges'][:-1])
    for v, c in (('base', C_BASE), ('lk', C_LK), ('sq', C_MOD)):
        m, lo, hi = band(s, 'pd', v)
        ax.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(x, m, color=c, lw=1.0)
    ax.plot(x, pooled_md(s, 'pd'), color=C_MD, lw=1.0)
    ax.plot(x, s['rnd_pd'], color=C_RND, lw=0.8, ls='--')
    ax.set_xlabel('pair distance (nm)')
    ax.set_ylabel('density (1/nm)')


def panel_big(ax, s, ks=None):
    k = np.arange(1, s['P'] + 1)
    ax.bar(k, pooled_md(s, 'big', ks), color='none', edgecolor=C_MD, lw=0.7, width=0.8)
    for v, c, d in (('base', C_BASE, -0.2), ('lk', C_LK, 0), ('sq', C_MOD, 0.2)):
        m, lo, hi = band(s, 'big', v, ks)
        ax.errorbar(k + d, m, yerr=[m - lo, hi - m], fmt='o', ms=2, color=c, lw=0.7, capsize=1.2)
    ax.plot(k, s['rnd_big'], 'x', ms=2.5, color=C_RND)
    ax.set_xlabel('largest cluster size')
    ax.set_ylabel('fraction of frames')
    ax.set_xticks(k if s['P'] <= 7 else k[1:10:2])
    if s['P'] > 7:
        ax.set_xlim(0.4, 10.6)


def panel_surv(ax, s):
    for v, c in (('base', C_BASE), ('lk', C_LK), ('sq', C_MOD)):
        m, lo, hi = band(s, 'surv', v)
        ax.fill_between(LAGS, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(LAGS, m, color=c, lw=1.0)
    ax.plot(LAGS, pooled_md(s, 'surv'), color=C_MD, lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel('lag (frames)')
    ax.set_ylabel('contact survival')


def model_handles(random=True, off=True):
    kw = dict(lw=1.0, ms=3, mew=0.7)
    h = [plt.Line2D([], [], color=C_MD, lw=1.0, label='MD'),
         plt.Line2D([], [], color=C_MOD, marker='o', label='full model', **kw),
         plt.Line2D([], [], color=C_OFF, marker='^', mfc='none', label='ablation', **kw),
         plt.Line2D([], [], color=C_LK, marker='s', mfc='none', label='backbone model', **kw),
         plt.Line2D([], [], color=C_BASE, marker='o', mfc='none', label='comparison model', **kw)]
    return [x for x in h if off or x.get_label() != 'ablation'] + ([plt.Line2D([], [], color=C_RND, lw=0.8, ls='--', label='random placement')] if random else [])


def finish(fig, handles, ncol=3, w_pad=0.3):
    """one shared legend at the top inside the figure, panels laid out below it"""
    rows = -(-len(handles) // ncol)
    top = 1 - (rows * 0.15 + 0.01) / fig.get_figheight()
    fig.legend(handles=handles, loc='upper center', ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8,
               handlelength=1.8, handletextpad=0.4, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, top), pad=0.3, w_pad=w_pad, h_pad=0.6)


def fig_clustering(S, T):
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.36), gridspec_kw={'width_ratios': [1, 1.5, 1.1]})
    s7, s14 = S['7-PIEZO'], S['14-PIEZO']
    dots(axes[0, 0], S, [('7-PIEZO', k) for k in s7['md']], lambda r, k: r['kl'][k]['logpair_kl_share'],
         lambda n, k: T[(n, k)]['replica']['logpair_kl_share'])
    axes[0, 0].set_ylabel(r'$\kappa$')
    slots = [(n, k) for n, s in S.items() for k in s['md']]
    dots(axes[0, 1], S, slots, lambda r, k: r['g'][k], lambda n, k: S[n]['rows']['sq'][0]['g_floor'][k])
    axes[0, 1].set_ylabel('histogram score')
    panel_pd(axes[0, 2], s7)
    panel_big(axes[1, 0], s7)
    panel_surv(axes[1, 1], s7)
    panel_pd(axes[1, 2], s14)
    for a in axes.ravel():
        style(a)
    for a, c in zip(axes.ravel(), 'abcdef'):
        letter(a, c)
    finish(fig, model_handles())
    save(fig, 'fig_clustering')


def fig_supp_clustering(S, T):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.26))
    slots = [(n, k) for n, s in S.items() for k in s['md']]
    dots(axes[0], S, slots, lambda r, k: r['kl'][k]['late_logpair_kl_share'], lambda n, k: T[(n, k)]['replica']['late_logpair_kl_share'])
    axes[0].set_ylabel(r'$\kappa$, frames $\geq$ 3,000')
    panel_big(axes[1], S['14-PIEZO'], list(S['14-PIEZO']['md'])[1:])
    panel_surv(axes[2], S['14-PIEZO'])
    for a in axes[1:]:
        style(a)
    for a, c in zip(axes, 'abc'):
        letter(a, c)
    finish(fig, model_handles())
    save(fig, 'fig_supp_clustering')


def kl_table(S):
    """per system and trajectory: the other second-half MD trajectory scored as a rollout (the MD floor), and the calibration
    KL(MD||random) / KL(MD||other MD) of the log pair distance (the κ floor is its inverse)"""
    def late(x, md, rnd, box, start=3000):
        lm = H.lnpd(md[::H.STRIDE], box).ravel()
        kr = H.gparts(lm, H.lnpd(rnd, box).ravel())[0]
        k, m, sp = H.gparts(lm, H.lnpd(x[:, start::H.STRIDE], box).ravel())
        return {'late_logpair_kl_share': k / kr, 'late_logpair_mean_share': m / kr, 'late_logpair_spread_share': sp / kr}

    out = {}
    for name, s in S.items():
        rnd = np.random.default_rng(0).random((8000, s['P'], 2)) * s['box']
        a, b = list(s['md'])
        for k, other in ((a, b), (b, a)):
            x = s['md'][other][None]
            rep = {**H.kl_metrics(x, s['md'][k], rnd, s['box']), **late(x, s['md'][k], rnd, s['box'])}
            lm = H.lnpd(s['md'][k][::H.STRIDE], s['box']).ravel()
            kr, ko = H.gparts(lm, H.lnpd(rnd, s['box']).ravel())[0], H.gparts(lm, H.lnpd(s['md'][other][::H.STRIDE], s['box']).ravel())[0]
            out[(name, k)] = {'replica': rep, 'kl_random': float(kr), 'kl_other_md': float(ko), 'calibration': float(kr / ko)}
    return out


def groups(s, T, name, field):
    """per trajectory k: arrays of the field for baseline, without squash, model (one value per run, in seed order)"""
    return {k: {v: np.array([r['kl'][k][field] for r in s['rows'][v]]) for v in ('base', 'lk', 'nosq', 'sq')} for k in s['md']}


def fig_summary(S, T):
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.9))
    keys = [(n, k) for n, s in S.items() for k in s['md']]
    xs = np.arange(len(keys))
    for ax, field in ((axes[0], 'logpair_kl_share'), (axes[1], 'late_logpair_kl_share')):
        for i, (n, k) in enumerate(keys):
            g = groups(S[n], T, n, field)[k]
            for v, c, mk, off, fill in (('base', C_BASE, 'o', -0.36, False), ('lk', C_LK, 's', -0.16, False), ('nosq', C_OFF, '^', 0.04, False), ('sq', C_MOD, 'o', 0.24, True)):
                y = g[v]
                ax.plot(i + off + np.linspace(-0.04, 0.04, len(y)), y, mk, ms=2.8, mfc=c if fill else 'none', mec=c, mew=0.8, alpha=0.8 if fill else 1)
                ax.plot([i + off - 0.07, i + off + 0.07], [y.mean()] * 2, color=c, lw=1.4)
            ax.plot(i + 0.42, T[(n, k)]['replica'][field], 'D', ms=3.5, color=C_MD)
        ax.axhline(1.0, color=C_RND, ls='--', lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([('1', '2')[i % 2] for i in xs], fontsize=7)
        ax.set_yscale('log')
        ax.set_ylabel(r'$\kappa$' if field == 'logpair_kl_share' else r'$\kappa$, frames $\geq$ 3,000', fontsize=7)
        for j, name in enumerate(S):
            ax.text(2 * j + 0.5, -0.27, name, transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
        style(ax)
    ax = axes[2]
    for j, (name, s) in enumerate(S.items()):
        b, l, o, q = (np.array([r['overlap'] for r in s['rows'][v]]) for v in ('base', 'lk', 'nosq', 'sq'))
        for xo, y, c, mk, fill in ((-0.36, b, C_BASE, 'o', False), (-0.12, l, C_LK, 's', False), (0.12, o, C_OFF, '^', False), (0.36, q, C_MOD, 'o', True)):
            ax.plot(j + xo + np.linspace(-0.07, 0.07, len(y)), y, mk, ms=3, mfc=c if fill else 'none', mec=c, mew=0.9, zorder=2)
        jo, jq = np.linspace(-0.07, 0.07, len(o)), np.linspace(-0.07, 0.07, len(q))
        for oi, qi, a, c in zip(o, q, jo, jq):
            ax.plot([j + 0.12 + a, j + 0.36 + c], [oi, qi], color=C_RND, lw=0.5, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(S), fontsize=7)
    ax.set_xlim(-0.5, 1.5)
    ax.set_yscale('log')
    ax.set_ylabel('overlap', fontsize=7)
    style(ax)
    for a, c in zip(axes, 'abc'):
        letter(a, c)
    h = [plt.Line2D([], [], ls='', marker='o', ms=3, mfc='none', mec=C_BASE, label='comparison model'),
         plt.Line2D([], [], ls='', marker='s', ms=3, mfc='none', mec=C_LK, label='backbone model'),
         plt.Line2D([], [], ls='', marker='^', ms=3, mfc='none', mec=C_OFF, label='ablation'),
         plt.Line2D([], [], ls='', marker='o', ms=3, color=C_MOD, label='full model'),
         plt.Line2D([], [], ls='', marker='D', ms=3.5, color=C_MD, label='other MD trajectory'),
         plt.Line2D([], [], color=C_RND, ls='--', lw=0.9, label='random placement')]
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    save(fig, 'fig_summary', True)


def fig_squash(S):
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.99), gridspec_kw={'width_ratios': [0.95, 1.15, 1.1, 1.1]})
    ax = axes[0]
    lo_, hi_ = np.inf, -np.inf
    for j, (name, s) in enumerate(S.items()):
        tr = [np.load(os.path.join(TRACES, p['tag'] + '.npz')) for p in s['picks']['sq']]
        e = tr[0]['edges']
        xc = 0.5 * (e[1:] + e[:-1])
        cnt, hsum = sum(t['cnt'] for t in tr), sum(t['hsum'] for t in tr)      # closing steps only: the factor is applied there
        ok = cnt > 50
        lo_, hi_ = min(lo_, xc[ok].min()), max(hi_, xc[ok].max())
        ax.plot(xc[ok], hsum[ok] / cnt[ok], color=C_MOD, ls=('-', '--')[j], lw=1.0, label=name)
    ax.axvline(1.0, color=C_RND, ls=':', lw=0.8)
    ax.set_xlim(lo_, min(hi_, 1.4))   # the factor is 1 beyond ~1.2 of the core
    ax.set_xlabel(r'distance / $r_\mathrm{core}$')
    ax.set_ylabel('contraction factor')
    ax = axes[1]
    for j, (name, s) in enumerate(S.items()):
        b, l, o, q = (np.array([r['overlap'] for r in s['rows'][v]]) for v in ('base', 'lk', 'nosq', 'sq'))
        for xo, y, c, mk, fill in ((-0.36, b, C_BASE, 'o', False), (-0.12, l, C_LK, 's', False), (0.12, o, C_OFF, '^', False), (0.36, q, C_MOD, 'o', True)):
            ax.plot(j + xo + np.linspace(-0.07, 0.07, len(y)), y, mk, ms=2.3, mfc=c if fill else 'none', mec=c, mew=0.7, zorder=2)
        for oi, qi, a, c in zip(o, q, np.linspace(-0.07, 0.07, len(o)), np.linspace(-0.07, 0.07, len(q))):
            ax.plot([j + 0.12 + a, j + 0.36 + c], [oi, qi], color=C_RND, lw=0.4, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['7-P', '14-P'])
    ax.set_xlabel('system')
    ax.set_xlim(-0.55, 1.55)
    ax.set_yscale('log')
    ax.set_ylabel('overlap')
    x = 0.5 * (SHORT_EDGES[1:] + SHORT_EDGES[:-1])
    for ax, (name, s) in zip(axes[2:], S.items()):
        for v, c in (('base', C_BASE), ('lk', C_LK), ('nosq', C_OFF)):
            ax.plot(x, band(s, 'pd_short', v)[0], color=c, lw=0.9)
        m, lo, hi = band(s, 'pd_short')
        ax.fill_between(x, lo, hi, color=C_MOD, alpha=0.2, lw=0)
        ax.plot(x, m, color=C_MOD, lw=1.0)
        ax.plot(x, pooled_md(s, 'pd_short'), color=C_MD, lw=1.0)
        ax.axvline(s['core'] * s['nm'], color=C_RND, ls=':', lw=0.8)
        ax.set_xlabel('distance (nm)')
        ax.set_ylabel('density (1/nm)')
        ax.set_title(name.replace('PIEZO', 'P'), fontsize=8, pad=2, loc='right')
    for a, c in zip(axes, 'abcd'):
        style(a)
        letter(a, c)
    finish(fig, model_handles(False))
    save(fig, 'fig_squash')


def fig_membrane(S):
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.86))
    i = 0
    for name, s in S.items():
        for key, edges, xl, yl in (('dep_iso', s['iso_edges'], 'isolation (nm)', 'node depth (nm)'),
                                   ('dep_hist', s['dep_edges'], 'node depth (nm)', 'density (1/nm)')):
            ax = axes[i]
            x = 0.5 * (edges[1:] + edges[:-1])
            for v, c in (('base', C_BASE), ('lk', C_LK), ('sq', C_MOD)):
                m, lo, hi = band(s, key, v)
                ax.fill_between(x, lo, hi, color=c, alpha=0.2, lw=0)
                ax.plot(x, m, color=c, lw=1.0)
            ax.plot(x, pooled_md(s, key), color=C_MD, lw=1.0)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(name, fontsize=8, pad=2, loc='right')
            style(ax)
            ax.locator_params(axis='x', nbins=2)
            letter(ax, 'abcd'[i])
            i += 1
    finish(fig, model_handles(False, False), 4)
    save(fig, 'fig_membrane')


def summary(S, T):
    js = {}
    for name, s in S.items():
        ref, ks = s['ref'], list(s['md'])
        sq = lambda p: {k: float(np.load(os.path.join(TRACES, p['tag'] + '.npz'))[k]) for k in ('frac', 'frac_contracted', 'frac_closing')}
        js[name] = {
            'rc_nm': s['rc'] * s['nm'], 'core_nm': s['core'] * s['nm'], 'box_nm': s['box'] * s['nm'],
            'md_len': {k: len(s['md'][k]) for k in ks},
            'md_mean_largest_cluster': {k: ref.big[k] for k in ks}, 'random_mean_largest_cluster': ref.rand_big,
            'md_overlap': {k: float((fo.pair_dist_xy(s['md'][k], s['box']) < s['core']).mean()) for k in ks},
            'md_depth_coupling': dict(ref.r), 'md_node_sd': dict(ref.node_sd), 'md_hyper_sd': dict(ref.hyper_sd),
            'md_node_depth_median_nm': {k: s['md_stats'][k]['dep_median'] for k in ks},
            'md_survival': {k: dict(zip(map(int, LAGS), map(float, s['md_stats'][k]['surv']))) for k in ks},
            'model_node_depth_median_nm': {v: {k: float(np.median([r['dep_median'] for r in s['roll_stats'][v][k]])) for k in ks} for v in VARIANTS},
            'model_survival_median': {v: {k: dict(zip(map(int, LAGS), map(float, np.nanmedian([r['surv'] for r in s['roll_stats'][v][k]], 0))))
                                          for k in ks} for v in VARIANTS},
            'model_mean_largest_cluster_pooled': {v: {k: float(np.mean([(np.arange(1, s['P'] + 1) * r['big']).sum() for r in s['roll_stats'][v][k]]))
                                                      for k in ks} for v in VARIANTS},
            'checkpoints': {v: [(r['run'], r['seed'], r['epoch'], r['n_epochs']) for r in s['rows'][v]] for v in VARIANTS},
            'squash_trace': {p['tag']: sq(p) for p in s['picks']['sq']},
            'kl_calibration': {k: {a: T[(name, k)][a] for a in ('kl_random', 'kl_other_md', 'calibration')} for k in ks},
            'kl_replica': {k: T[(name, k)]['replica'] for k in ks},
        }
    return js


if __name__ == '__main__':
    which = sys.argv[1:] or ['clustering', 'supp', 'summary', 'squash', 'membrane']
    S = {name: collect(name) for name in SYSTEMS}
    T = kl_table(S)
    for n, f in (('squash', fig_squash), ('membrane', fig_membrane)):
        if n in which:
            f(S)
    for n, f in (('clustering', fig_clustering), ('supp', fig_supp_clustering)):
        if n in which:
            f(S, T)
    if 'summary' in which:
        fig_summary(S, T)
    json.dump(summary(S, T), open(os.path.join(RESULTS, 'stats_half.json'), 'w'), indent=1, default=float)
