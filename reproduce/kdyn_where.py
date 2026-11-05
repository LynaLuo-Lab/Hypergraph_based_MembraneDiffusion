"""Where the change-term excess sits on 7-PIEZO: sd ratio by pair-distance bin (A), displacement correlation (B), K without core pairs (C).
usage: kdyn_where.py   (writes results/kdyn_where.txt)"""
import os, sys
import numpy as np
from scipy import stats as sps
import common
from common import ROOT, SEED, picks, reference
from paths import RESULTS, ROLLS
H = common.heldout_lk()
os.chdir(ROOT)
md, depth, thetas, box = reference('7-PIEZO')
ks = list(md)
RC, CT = 53.2, 171.55
EDG = [0, RC, CT, 2 * CT, np.inf]
BN = ['<core', 'core-ct', 'ct-2ct', '>2ct']
I, J = np.triu_indices(7, 1)
def mi(d): return d - box * np.round(d / box)
def rdist(x):
    return np.sqrt((mi(x[..., I, :2] - x[..., J, :2]) ** 2).sum(-1))
def binid(r): return np.digitize(r, EDG[1:-1])
def cum(x):  # unwrapped cumulative position [..., T, P, 2]
    s = mi(x[..., 1:, :, :2] - x[..., :-1, :, :2])
    return np.concatenate([np.zeros_like(s[..., :1, :, :]), np.cumsum(s, -3)], -3)

def partA(x, L):
    r = rdist(x); l = np.log(np.maximum(r, 1.0))
    d = (l[..., L:, :] - l[..., :-L, :]).ravel(); b = binid(r[..., :-L, :]).ravel()
    tot = ((d - d.mean()) ** 2).sum()
    return [(np.mean(b == k), d[b == k].std(), ((d[b == k] - d.mean()) ** 2).sum() / tot) for k in range(4)]

def partB(x, L):
    r = rdist(x); b = binid(r[..., :-L, :]).ravel()
    c = cum(x); dx = c[..., L:, :, :] - c[..., :-L, :, :]
    out = {}
    rms = np.sqrt((dx ** 2).sum(-1).mean())
    for nm, dd in (('raw', dx), ('dem', dx - dx.mean(-2, keepdims=True))):
        dot = (dd[..., I, :] * dd[..., J, :]).sum(-1).ravel()
        n2 = 0.5 * ((dd[..., I, :] ** 2).sum(-1) + (dd[..., J, :] ** 2).sum(-1)).ravel()
        out[nm] = [dot[b == k].sum() / n2[b == k].sum() for k in range(4)]
    return out, rms

def partC(lm, rm, lx, rx, L):
    res = {}
    for ex in (0, 1):
        dm = lm[L:] - lm[:-L]; dx = lx[..., L:, :] - lx[..., :-L, :]
        if ex:
            km = (rm[L:] >= RC) & (rm[:-L] >= RC); kx = (rx[..., L:, :] >= RC) & (rx[..., :-L, :] >= RC)
            dm, dx = dm[km], dx[kx]
        else: dm, dx = dm.ravel(), dx.ravel()
        res[ex] = (H.gparts(dm, dx)[0], sps.kurtosis(dx))
    res['kmd'] = {ex: sps.kurtosis((lm[L:] - lm[:-L])[((rm[L:] >= RC) & (rm[:-L] >= RC))] if ex else (lm[L:] - lm[:-L]).ravel()) for ex in (0, 1)}
    res['fr'] = float(1 - ((rx[..., L:, :] >= RC) & (rx[..., :-L, :] >= RC)).mean())
    return res

VN = {'sq': 'full', 'lk': 'backbone', 'base': 'comparison'}
runs = {v: [] for v in VN}
for v in VN:
    for p in sorted(picks('7-PIEZO', (v,)), key=lambda p: p['seed']):
        runs[v].append(np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz")))
out = []; w = out.append
ms = lambda a, f='.3f': f'{np.mean(a):{f}}[{np.min(a):{f}},{np.max(a):{f}}]'
for k in ks:
    w(f'\n===== {k} (box {box:.1f}) =====')
    xm = md[k][:, :, :2]
    for L in (10, 100):
        mA = partA(xm, L)
        rA = np.array([partA(z[f'xy_{k}'][..., :2], L) for z in runs['sq']])  # runs,4,3
        w(f'-- A lag {L}: bin | frac model/MD | sd model[min,max] / MD | ratio | var share model / MD')
        for i in range(4):
            w(f'  {BN[i]:8s} {rA[:,i,0].mean():.3f}/{mA[i][0]:.3f} | {ms(rA[:,i,1],".4f")} / {mA[i][1]:.4f} | {ms(rA[:,i,1]/mA[i][1],".2f")} | {ms(rA[:,i,2],".3f")} / {mA[i][2]:.3f}')
        mB, mr = partB(xm, L)
        rB = [partB(z[f'xy_{k}'][..., :2], L) for z in runs['sq']]
        w(f'-- B lag {L}: C per bin (model mean[min,max] / MD); rms disp ratio {ms([b[1]/mr for b in rB],".3f")}')
        for nm in ('raw', 'dem'):
            for i in range(4):
                w(f'  {nm} {BN[i]:8s} {ms([b[0][nm][i] for b in rB])} / {mB[nm][i]:.3f}')
    lm = H.lnpd(xm, box); rm = rdist(xm)
    o = [j for j in ks if j != k][0]; xo = md[o][None, :, :, :2]
    for L in (1, 10):
        w(f'-- C lag {L}: model | K full-data -> excl core | kurt -> excl | frac excluded  (MD kurt {{}})')
        for v in list(VN) + ['floor']:
            xs = [xo] if v == 'floor' else [z[f'xy_{k}'][..., :2] for z in runs[v]]
            rs = [partC(lm, rm, H.lnpd(x, box), rdist(x), L) for x in xs]
            f = lambda g: ms([g(r) for r in rs], '.4f')
            w(f'  {v:10s} K {f(lambda r: r[0][0])} -> {f(lambda r: r[1][0])} | kurt {f(lambda r: r[0][1])} -> {f(lambda r: r[1][1])} | excl {np.mean([r["fr"] for r in rs]):.4f}')
        w(f'  MD kurt {rs[0]["kmd"][0]:.2f} -> {rs[0]["kmd"][1]:.2f}')
txt = '\n'.join(out); print(txt)
open(os.path.join(RESULTS, 'kdyn_where.txt'), 'w').write(txt + '\n')
