"""Snapshots of the held-out trajectory 2 videos (first frame, 25%, 75% and the last frame), MD against the full model's best rollout,
for both systems: same rollouts, one-time shift, circles (diameter r_core, depth under the protein), hyperedge junction dots, contact
lines and symmetric RdBu limits as video.py. Frames: 1, then for the targets round(0.25 T), round(0.75 T) and T (T the last
frame) the nearest frame, over every frame, where MD's and the model's largest clusters are equal (ties to the earlier frame; for T only
backward from the last frame). Writes figures/fig_snapshots.pdf (and .png).
usage: fig_snapshots.py [top]   (top: the frame number is a title above every panel, written to fig_snapshots_top; default: inside each panel)
In both, 'largest cluster: n' is a boxed label in a panel corner that holds no protein circle, the same corner for MD and model of a column."""
import sys
import os

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import Normalize

from figures import fo, plt, save, setup
from paths import HYPER
from video import JOBS, SH, STRIDE, contacts, junctions

PW, GAPX, GAPY, GAPS = 0.94, 0.04, 0.04, 0.12          # panel side, gaps (inches)
LM, CB, CPAD, RM, BM = 0.30, 0.08, 0.05, 0.42, 0.05
TITLE = 0.17                                            # room for a frame title above a panel (top version), inches
LAB_H = 0.11                                            # height of an in-panel label box, inches
DOT = 0.3                                               # junction dot diameter / circle diameter
cmap = plt.get_cmap('RdBu').copy()
cmap.set_bad('0.6')


def system(name, key, tag):
    s = setup(name)
    nm, L, P = s['nm'], s['box'] * s['nm'], s['P']
    z = np.load(os.path.join(HYPER, tag + '_hyper.npz'))
    xr, dr, hr, mem = z['xy'], z['node'], z['hyper'], z['members'].astype(bool)
    xm, dm, hm = s['md'][key][..., :2], s['depth'][key]['node'], s['depth'][key]['hyper']
    n = len(xm)
    last = n - 1
    bm, br = (fo.largest_cluster(fo.pair_dist_xy(x, s['box']), s['rc']) for x in (xm, xr))
    fr, tg = [1], [int(round(0.25 * last)), int(round(0.75 * last)), last]
    for i, tt in enumerate(tg):
        fr.append(next(j for k in range(last) for j in ((tt - k, tt + k) if i < 2 else (tt - k,)) if 1 <= j <= last and bm[j] == br[j]))
    print(f'{name}: targets {tg}, chosen {fr[1:]}, offsets {[f - t_ for f, t_ in zip(fr[1:], tg)]} ({[round(abs(f - t_) / last * 100, 2) for f, t_ in zip(fr[1:], tg)]} % of T), '
          f'largest cluster MD {[int(bm[f]) for f in fr]} model {[int(br[f]) for f in fr]}', flush=True)
    a = 2 * np.pi * np.mod(xm[0] * nm, L) / L
    sh = L / 2 - np.mod(np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0)), 2 * np.pi) * L / (2 * np.pi)
    vf = np.arange(1, n, STRIDE)                       # the video's frames set the colour limits
    lim = np.nanpercentile(np.abs(np.concatenate([(dm[vf] * nm).ravel(), (dr[vf - 1] * nm).ravel(), (hm[vf] * nm).ravel(), (hr[vf - 1] * nm).ravel()])), 99)
    Hm = mem / mem.sum(1, keepdims=True)
    rows = []
    for x, d, h, off in ((xm, dm, hm, 0), (xr, dr, hr, 1)):
        rows.append([dict(p=np.mod(x[f] * nm + sh, L), d=d[f - off] * nm, h=h[f - off] * nm,
                          big=int(fo.largest_cluster(fo.pair_dist_xy(x[f], s['box']), s['rc']))) for f in fr])
    d0 = rows[0][0]['p'] - rows[1][0]['p']
    d0 -= L * np.round(d0 / L)
    print(f'{name}: frames {fr}, L {lim:.2f} nm, box {L:.1f} nm, max MD-model position difference at frame 1 {np.abs(d0).max():.2f} nm, '
          f'clusters {[[c["big"] for c in r] for r in rows]}', flush=True)
    return dict(s=s, L=L, nm=nm, lim=lim, fr=fr, rows=rows, Hm=Hm, D=s['core'] * nm, rc=s['rc'] * nm)


def box(S, hx, hy, lv, w):
    """(x0, x1, y0, y1) in data units of a label of width w inches in a corner (hx: 0 left, 1 right; hy: 0 bottom, 1 top), lv rows in from the edge"""
    L = S['L']
    bw, bh, st = w / PW * L, LAB_H / PW * L, (LAB_H + 0.02) / PW * L
    x0 = 0.03 * L if hx == 0 else 0.97 * L - bw
    y0 = 0.03 * L + lv * st if hy == 0 else 0.97 * L - bh - lv * st
    return x0, x0 + bw, y0, y0 + bh


def corners(S, cs, first, wf, wc, top):
    """(frame slot, cluster slot) of a column, a slot being (hx, hy, rows in): the first pair, in a fixed order, whose label boxes overlap
    neither each other, nor a protein circle (any periodic image, either row), nor the scale bar (first column); with frame titles above
    the panels only the cluster label is inside"""
    sl = [(hx, hy, lv) for lv in (0, 1, 2) for hx, hy in ((0, 1), (1, 1), (0, 0), (1, 0))]
    L, r = S['L'], S['D'] / 2
    sb = (0.03 * L, 0.06 * L + 100 + 0.02 * L, 0.03 * L, 0.2 * L)
    img = [np.concatenate([c['p'] + L * np.array(o) for o in SH]) for c in cs]
    hit = lambda b, i: int((np.hypot(np.maximum(np.maximum(b[0] - i[:, 0], i[:, 0] - b[1]), 0), np.maximum(np.maximum(b[2] - i[:, 1], i[:, 1] - b[3]), 0)) < r).sum())
    meet = lambda a, b: a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]
    best = None
    for fc in ((None,) if top else sl):
        for cc in sl:
            bs = ([] if fc is None else [box(S, *fc, wf)]) + [box(S, *cc, wc)]
            if (len(bs) == 2 and meet(*bs)) or (first and any(meet(b, sb) for b in bs)):
                continue
            n = sum(hit(b, i) for b in bs for i in img)
            if n == 0:
                return fc, cc
            if best is None or n < best[0]:
                best = (n, fc, cc)
    print(f'no slot free of protein circles for frames {S["fr"]}: least overlap {best[0]} circle(s) over both rows under a label', flush=True)
    return best[1], best[2]


def label(ax, S, hx, hy, lv, w, text):
    x0, x1, y0, y1 = box(S, hx, hy, lv, w)
    L = S['L']
    ax.text((x0 if hx == 0 else x1) / L, (y0 if hy == 0 else y1) / L, text, transform=ax.transAxes, ha='left' if hx == 0 else 'right',
            va='bottom' if hy == 0 else 'top', fontsize=5.5, zorder=7, bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))


def panel(ax, S, c, first, frame, fc, cc, wf, wc, top=False):
    L, D = S['L'], S['D']
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.add_collection(LineCollection(contacts(c['p'], L, S['rc']), colors='0.45', linewidths=0.5, zorder=2))
    cen = junctions(c['p'], S['Hm'], L)
    for pos, dep, dia, ec, lw, zo in ((cen, c['h'], DOT * D, '0.35', 0.15, 2.5), (c['p'], c['d'], D, '0.15', 0.4, 3)):
        e = EllipseCollection(np.full(9 * len(pos), dia), np.full(9 * len(pos), dia), np.zeros(9 * len(pos)), units='xy',
                              offsets=np.concatenate([pos + L * np.array(o) for o in SH]), offset_transform=ax.transData,
                              cmap=cmap, edgecolors=ec, linewidths=lw, zorder=zo)
        e.set_array(np.ma.masked_invalid(np.tile(dep, 9)))
        e.set_clim(-S['lim'], S['lim'])
        ax.add_collection(e)
    if top:
        ax.set_title(f'frame {frame}', fontsize=8, pad=3)
    else:
        label(ax, S, *fc, wf, f'frame {frame}')
    label(ax, S, *cc, wc, f"largest cluster: {c['big']}")
    if first:
        ax.plot([0.06 * L, 0.06 * L + 100], [0.05 * L] * 2, color='black', lw=1.2, solid_capstyle='butt', zorder=6)
        ax.text(0.06 * L + 50, 0.07 * L, '100 nm', ha='center', va='bottom', fontsize=6, zorder=6)


def main(top=False):
    SS = [system(name, key, tag) for name, key, tag in (JOBS['7'], JOBS['14'])]
    gy, gs, tm = GAPY + (TITLE if top else 0), GAPS + (TITLE if top else 0), 0.05 + (TITLE if top else 0)
    wid = LM + 4 * PW + 3 * GAPX + CPAD + CB + RM
    hei = tm + 4 * PW + 2 * gy + gs + BM
    fig = plt.figure(figsize=(wid, hei))
    meas = [fig.text(0, 0, s, fontsize=5.5) for s in ('largest cluster: 6', 'frame 5675')]
    r = fig.canvas.get_renderer()
    wc, wf = (m.get_window_extent(r).width / fig.dpi + 0.06 for m in meas)
    for m in meas:
        m.remove()
    ytop = hei - tm
    for si, (S, lab) in enumerate(zip(SS, ('7-P', '14-P'))):
        yb = ytop - si * (2 * PW + gy + gs)
        for ci in range(4):
            fc, cc = corners(S, [S['rows'][0][ci], S['rows'][1][ci]], ci == 0, wf, wc, top)
            for ri in range(2):
                ax = fig.add_axes([(LM + ci * (PW + GAPX)) / wid, (yb - ri * (PW + gy) - PW) / hei, PW / wid, PW / hei])
                panel(ax, S, S['rows'][ri][ci], ci == 0 and ri == 0, S['fr'][ci], fc, cc, wf, wc, top)
                if ci == 0:
                    ax.text(-0.08, 0.5, f"{lab} {('MD', 'model')[ri]}", transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=8)
        cax = fig.add_axes([(LM + 4 * PW + 3 * GAPX + CPAD) / wid, (yb - 2 * PW - gy) / hei, CB / wid, (2 * PW + gy) / hei])
        cb = fig.colorbar(ScalarMappable(Normalize(-S['lim'], S['lim']), cmap), cax=cax)
        cb.set_label('depth (nm)', fontsize=8, labelpad=2)
        cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    print(f'figure {wid * 2.54:.2f} x {hei * 2.54:.2f} cm before the tight crop; circle d = {SS[0]["D"]:.1f} nm, dot d = {DOT} x circle', flush=True)
    save(fig, 'fig_snapshots_top' if top else 'fig_snapshots')


if __name__ == '__main__':
    main('top' in sys.argv[1:])
