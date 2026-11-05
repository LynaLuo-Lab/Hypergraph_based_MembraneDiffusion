"""Change in ln pair distance over lags 1, 10, 100, 1000 on 7-PIEZO: K split into mean and spread parts, sd ratio and kurtosis per model.
usage: kdyn_split.py   (writes results/kdyn_split.txt)"""
import os, sys, json
import numpy as np
from scipy import stats as sps
import common
from common import ROOT, SEED, picks, reference
from paths import RESULTS, ROLLS
H = common.heldout_lk()
os.chdir(ROOT)
LAGS = H.DLAGS
md, depth, thetas, box = reference('7-PIEZO')
ks = list(md)
print('box', box, 'keys', ks, {k: md[k].shape for k in ks})
lm = {k: H.lnpd(md[k], box) for k in ks}
rnd = np.random.default_rng(0).random((8000, md[ks[0]].shape[1], 2)) * box

def chg(l, L):  # l [..., T, pairs]
    return (l[..., L:, :] - l[..., :-L, :]).ravel()

def stat(dm, dx):
    k, m, s = H.gparts(dm, dx)
    return dict(K=float(k), Km=float(m), Ks=float(s), sdr=float(dx.std() / dm.std()), mean=float(dx.mean()), sd=float(dx.std()),
                kx=float(sps.kurtosis(dx)), km=float(sps.kurtosis(dm)))
MDch = {(k, L): chg(lm[k], L) for k in ks for L in LAGS}
res = {}
VN = {'sq': 'full', 'lk': 'backbone', 'base': 'comparison'}
for v in VN:
    P = picks('7-PIEZO', (v,))
    P = sorted(P, key=lambda p: p['seed'])
    print(v, len(P), flush=True)
    for p in P:
        z = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
        for k in ks:
            lx = H.lnpd(z[f'xy_{k}'], box)
            for L in LAGS:
                res[(v, p['tag'], k, L)] = stat(MDch[(k, L)], chg(lx, L))
# MD floor
for k in ks:
    o = [j for j in ks if j != k][0]
    lo = lm[o][None]
    for L in LAGS:
        res[('floor', 'floor', k, L)] = stat(MDch[(k, L)], chg(lo, L))
# static
stat_ = {}
for v in ('sq',):
    for p in sorted(picks('7-PIEZO', (v,)), key=lambda p: p['seed']):
        z = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
        for k in ks:
            r = H.kl_metrics(z[f'xy_{k}'], md[k], rnd, box)
            lmd = H.lnpd(md[k][::H.STRIDE], box).ravel()
            kr = H.gparts(lmd, H.lnpd(rnd, box).ravel())[0]
            kraw = r['logpair_kl_share'] * kr
            stat_[(p['tag'], k)] = dict(Kraw=float(kraw), Krand=float(kr), share=r['logpair_kl_share'], dK=r['dlogpair_kl'])
out = []
w = out.append
def agg(v, k, L, f):
    a = np.array([r[f] for (vv, t, kk, LL), r in res.items() if vv == v and kk == k and LL == L])
    return a
for k in ks:
    w(f'\n===== {k} =====')
    w('reproduction: mean over lags of K (mean over runs, [min,max]) ; also direct dlogpair_kl from kl_metrics for full model')
    for v in list(VN) + ['floor']:
        per = []
        runs = sorted({t for (vv, t, kk, LL) in res if vv == v and kk == k})
        for t in runs:
            per.append(np.mean([res[(v, t, k, L)]['K'] for L in LAGS]))
        w(f'  {v}: Kdyn {np.mean(per):.4f} [{min(per):.4f},{max(per):.4f}] n={len(per)}')
    sk = [d['dK'] for (t, kk), d in stat_.items() if kk == k]
    w(f'  full dK direct from kl_metrics: {np.mean(sk):.4f}')
    w('lag | model | sdratio mean[min,max] | K | Kmean | Kspread | kurt_model | kurt_MD | mean_d model | mean_d MD | sd model | sd MD')
    for L in LAGS:
        for v in list(VN) + ['floor']:
            f = lambda n: agg(v, k, L, n)
            def s(n, d=3):
                a = f(n); return f'{a.mean():.{d}f}[{a.min():.{d}f},{a.max():.{d}f}]'
            w(f'{L:5d} | {v:10s} | sdr {s("sdr")} | K {s("K",4)} | Km {s("Km",4)} | Ks {s("Ks",4)} | kurt {s("kx",2)} | kurtMD {f("km").mean():.2f} | mean {f("mean").mean():+.5f} / MD {np.mean([MDch[(k,L)].mean()]):+.5f} | sd {s("sd",4)} | sdMD {MDch[(k,L)].std():.4f}')
w('\nstatic (full model): raw Gaussian K of lnpd (stride 4), K_rand, share')
for k in ks:
    a = [d for (t, kk), d in stat_.items() if kk == k]
    w(f'{k}: Kraw {np.mean([d["Kraw"] for d in a]):.5f} [{min(d["Kraw"] for d in a):.5f},{max(d["Kraw"] for d in a):.5f}]  K_rand {a[0]["Krand"]:.4f}  share {np.mean([d["share"] for d in a]):.4f}')
txt = '\n'.join(out)
print(txt)
open(os.path.join(RESULTS, 'kdyn_split.txt'), 'w').write(txt + '\n')
