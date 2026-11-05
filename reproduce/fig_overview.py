"""Overview of the membrane simulator for 7-PIEZO: the MD frame, the hypergraph membrane module, the motion model, the squash
module and the next frame, with the loop back to the module. Panels a and b-e draw the first frame of trajectory 1's second
half (MD mesh frame 4262, ts 83350000) and one held-out rollout from it. Writes figures/fig_overview.pdf (and .png).
usage: fig_overview.py"""
import json
import os

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from figures import OUT, fo, letter, plt
from membrane.FitObjectiveCallback import load_md
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull

from paths import MD_DIR, MESH_FRAME, ROLLS, ROOT

W = 12.0 / 2.54
H = 3.65 / 2.54
NM = 0.4018202
CFG = 'configs/7piezo/full.json'
ARMS = os.path.join(MD_DIR, 'membrane_train_7piezo_R1second_part0_arms.npz')
ROLL = os.path.join(ROLLS, 'h7_lk_both_sq_s705_sq_e037_seed3000.npz')
KEY, R, MAG, TRACE = '7piezo_R1second', 0, 100, 1500
CORE = 21.4
PW, GAP, X0, Y0 = 1.85 / 2.54, 0.6 / 2.54, 0.25 / 2.54, 1.0 / 2.54
C_STEP = '#c0392b'


def junctions(xy, box, rc):
    """pairs and triples of proteins all within rc of each other, with their periodic centroids"""
    P = len(xy)
    d = fo.pair_dist_xy(xy[None], box)[0]
    D = np.zeros((P, P))
    i, j = np.triu_indices(P, 1)
    D[i, j] = D[j, i] = d
    ang = xy / box * 2 * np.pi
    cen = lambda m: (np.arctan2(np.sin(ang[m]).mean(0), np.cos(ang[m]).mean(0)) % (2 * np.pi)) / (2 * np.pi) * box
    pairs = [(a, b) for a, b in zip(i, j) if D[a, b] <= rc]
    tri = [(a, b, c) for a in range(P) for b in range(a + 1, P) for c in range(b + 1, P)
           if max(D[a, b], D[a, c], D[b, c]) <= rc]
    return pairs, tri, cen


def to_fig(x, y):
    return x / W, y / H


def panel(i):
    ax = plt.gcf().add_axes([(X0 + i * (PW + GAP)) / W, Y0 / H, PW / W, PW / H])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    return ax


def arrow_between(fig, i, label=None):
    x0 = X0 + i * (PW + GAP) + PW + 0.03
    x1 = x0 + GAP - 0.06
    y = Y0 + PW / 2
    fig.add_artist(FancyArrowPatch((x0 / W, y / H), (x1 / W, y / H), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=6, lw=0.7, color='black'))
    if label:
        fig.text((x0 + x1) / 2 / W, (y + 0.1 / 2.54) / H, label, ha='center', va='bottom', fontsize=7, linespacing=1.0)


def protein_marks(ax, p, theta, ms=3.2, lw=0.6, mfc='none'):
    ax.plot(p[:, 0], p[:, 1], 'o', ms=ms, mfc=mfc, mec='black', mew=lw, zorder=5)
    if theta is not None:
        for k in range(3):
            a = theta + k * 2 * np.pi / 3
            u = np.stack([np.cos(a), np.sin(a)], -1)
            for q, v in zip(p, u):
                ax.plot([q[0] + 6 * v[0], q[0] + 24 * v[0]], [q[1] + 6 * v[1], q[1] + 24 * v[1]], color='black', lw=0.6, zorder=5,
                        solid_capstyle='butt')


def junction_marks(ax, xy, box, rc, to_nm, star=False):
    pairs, tri, cen = junctions(xy, box, rc)
    for m in tri + pairs:
        c = to_nm(cen(list(m)))
        if star:
            for k in m:
                ax.plot(*zip(c, to_nm(xy[k])), color='0.2', lw=0.6, zorder=2)
        if len(m) == 3:
            ax.plot(*c, '^', ms=3, color='black', mec='white', mew=0.3, zorder=4)
        else:
            ax.plot(*c, 'o', ms=2.2, color='white', mec='black', mew=0.4, zorder=3)


def hull(pts, pad):
    """convex hull of the members, grown by pad: the hull of circles of radius pad around each member"""
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    c = (np.asarray(pts)[:, None, :] + pad * np.stack([np.cos(a), np.sin(a)], -1)[None]).reshape(-1, 2)
    return c[ConvexHull(c).vertices]


def main():
    cfg = json.load(open(os.path.join(ROOT, CFG)))
    box, rc = float(cfg['box_size']), float(cfg.get('fit_rc') or fo.RC)
    mesh = np.load(MESH_FRAME)
    zl = np.array(mesh['zl'], dtype=np.float64)
    zl = (zl - np.nanmean(zl)) / 10.0
    n, org = zl.shape[0], float(mesh['origin_A']) / 10.0
    L = box * NM
    ext = (-L / 2, L / 2)
    to_nm = lambda p: (np.asarray(p) * NM + L / 2) % L - L / 2
    xs = ext[0] + (np.arange(300) + 0.5) / 300 * L
    u = ((xs - org) / (L / n)) % n
    img = map_coordinates(zl, [u[None, :] + 0 * u[:, None], u[:, None] + 0 * u[None, :]], order=1, mode='grid-wrap')

    md = load_md([os.path.join(MD_DIR, '7piezo_R1second.csv')])[KEY]
    arms = np.load(ARMS, allow_pickle=True)
    assert np.abs(arms['coords'][0].reshape(-1) - md[0].reshape(-1)).max() < 1e-2
    theta = arms['theta'][0]
    xy0 = md[0][:, :2]
    roll = np.load(ROLL)
    rolls = roll['xy_' + KEY]
    assert np.abs(rolls[:, 0] - xy0).max() < 1e-3
    p = to_nm(xy0)
    dxs = rolls[:, 1] - rolls[:, 0]
    dxs -= box * np.round(dxs / box)
    stp = np.hypot(dxs[..., 0], dxs[..., 1]) * NM
    quiet = np.array([fo.pair_dist_xy(r[:2], box).min() > 2.0 * CORE / NM for r in rolls])
    for MAG in range(100, 0, -1):
        ok = [r for r in range(len(rolls)) if quiet[r] and np.abs(p + dxs[r] * NM * MAG).max() <= L / 2 - 10]
        if ok:
            break
    R = max(ok, key=lambda r: stp[r].min())
    print('rollout', R, 'MAG', MAG, 'step min/max (nm)', round(stp[R].min(), 2), round(stp[R].max(), 2), 'candidates', len(ok))
    xy = rolls[R]
    dx = dxs[R]
    dep = roll['node_depth_' + KEY][R, 0] * NM      # npz units (4.018 A) -> nm; index 0 is rollout frame 1
    assert np.abs(p + dx * NM * MAG).max() <= L / 2 - 10, 'arrowhead outside'

    pn = map_coordinates(zl, [((to_nm(xy0)[:, 0] - org) / (L / n)) % n, ((to_nm(xy0)[:, 1] - org) / (L / n)) % n], order=1, mode='grid-wrap')
    mdd = arms['node_depth'][0] * NM
    print('node depth at the 7 proteins (nm), MD npz:', np.round(mdd, 1), 'mesh:', np.round(pn, 1), 'max diff', np.abs(mdd - pn).max())
    assert np.abs(mdd - pn).max() < 2
    lim = 40
    print('max |field| (nm)', round(np.abs(zl).max(), 1), 'max |model node depth| (nm)', round(np.abs(dep).max(), 1))
    fig = plt.figure(figsize=(W, H))
    kw = dict(extent=(*ext, *ext), origin='lower')
    def lab(ax, t, c):
        letter(ax, c)
        ax.texts[-1].set_position((-0.02, 1.03))
        ax.text(0.12, 1.03, t, transform=ax.transAxes, fontsize=7, va='bottom', linespacing=1.0)

    ax = panel(0)
    ax_a = ax
    im = ax.imshow(img, cmap='RdBu', vmin=-lim, vmax=lim, interpolation='bilinear', **kw)
    junction_marks(ax, xy0, box, rc, to_nm)
    protein_marks(ax, to_nm(xy0), theta)
    ax.set_xlim(*ext)
    ax.set_ylim(*ext)
    ax.plot([ext[0] + 20, ext[0] + 120], [ext[0] + 22] * 2, color='black', lw=1.2, solid_capstyle='butt', zorder=6)
    ax.text(ext[0] + 15, ext[0] + 32, '100 nm', ha='left', va='bottom', fontsize=7, zorder=6,
            bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.7))
    lab(ax, 'MD frame', 'a')
    cax = fig.add_axes([X0 / W, (Y0 - 0.32 / 2.54) / H, PW / W, 0.09 / 2.54 / H])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cb.set_label('depth (nm)', fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.locator = plt.MaxNLocator(3)
    cb.update_ticks()
    arrow_between(fig, 0, '$\\mathbf{x}_i^0$')

    ax = panel(1)
    ax_b = ax
    grp = [3, 4, 5, 6]
    d = xy0[grp] - xy0[4]
    d -= box * np.round(d / box)
    cl = to_nm(xy0[4]) + d * NM
    pairs, tri, cen = junctions(xy0, box, rc)
    assert {(3, 4), (4, 6), (5, 6)} <= set(map(tuple, pairs)), pairs
    TRI = (3, 4, 6)
    cc = (cl.min(0) + cl.max(0)) / 2
    hh = (cl.max(0) - cl.min(0)).max() / 2 + 20
    pad = 0.2 * hh
    ax.set_xlim(cc[0] - hh, cc[0] + hh)
    ax.set_ylim(cc[1] - hh, cc[1] + hh)
    P_ = dict(zip(grp, cl))
    edges = [((3, 4, 5, 6), '#e4e0d6', '#8a8470'), ((3, 4, 6), '#d3dcd0', '#6f8a69'), ((5, 6), '#e6d6c4', '#9a7a55')]
    for m, fc, ec in edges:
        ax.add_patch(Polygon(hull([P_[k] for k in m], pad), closed=True, fc=fc, ec=ec, lw=0.5, alpha=0.8, zorder=1))
    jc = {}
    for m, fc, ec in edges:
        jc[m] = np.mean([P_[k] for k in m], 0)
        for k in m:
            ax.plot(*zip(jc[m], P_[k]), color='0.3', lw=0.6, zorder=2)
    C_MSG = '#a0452c'
    others = [jc[m] for m in jc if m != TRI]
    def arc_pts(a_, b_, rad):
        ctl = (a_ + b_) / 2 + rad * np.array([(b_ - a_)[1], -(b_ - a_)[0]])
        tt = np.linspace(0, 1, 60)[:, None]
        return (1 - tt) ** 2 * a_ + 2 * (1 - tt) * tt * ctl + tt ** 2 * b_
    rads = {}
    for k in TRI:
        rads[k] = max((0.3, -0.3, 0.4, -0.4, 0.25, -0.25), key=lambda r: min(
            np.linalg.norm(arc_pts(a_, b_, r) - o, axis=1).min() for a_, b_ in ((P_[k], jc[TRI]), (jc[TRI], P_[k])) for o in others))
        for a_, b_ in ((P_[k], jc[TRI]), (jc[TRI], P_[k])):
            ax.annotate('', xy=tuple(b_), xytext=tuple(a_), zorder=3,
                        arrowprops=dict(arrowstyle='-|>', lw=0.5, color=C_MSG, mutation_scale=4, shrinkA=3.5 if a_ is P_[k] else 1.5,
                                        shrinkB=1.5 if a_ is P_[k] else 3.5, connectionstyle=f'arc3,rad={rads[k]}'))
    for m in jc:
        print('junction', m, 'to members (nm):', [round(float(np.linalg.norm(jc[m] - P_[k])), 1) for k in m])
    print('arc rads', rads, 'min arc distance to other junctions (nm)', round(min(np.linalg.norm(arc_pts(a_, b_, rads[k]) - o, axis=1).min() for k in TRI for a_, b_ in ((P_[k], jc[TRI]), (jc[TRI], P_[k])) for o in others), 1))
    for m in jc:
        ax.plot(*jc[m], 'o', ms=2.6, mfc='white', mec='black', mew=0.5, zorder=4)
    ax.plot(cl[:, 0], cl[:, 1], 'o', ms=4, mfc='white', mec='black', mew=0.7, zorder=6)
    lab(ax, 'Hypergraph', 'b')
    arrow_between(fig, 1, '$\\mathbf{c}^t$')
    ax_a.add_patch(Rectangle((cc[0] - hh, cc[1] - hh), 2 * hh, 2 * hh, fill=False, ls='--', ec='0.3', lw=0.6, zorder=7))
    print('zoom window in panel a (nm): centre', np.round(cc, 1), 'half width', round(hh, 1))

    ax = panel(2)
    ax_c = ax
    ax.set_xlim(*ext)
    ax.set_ylim(*ext)
    ax.plot(p[:, 0], p[:, 1], 'o', ms=3.2, mfc='none', mec='0.5', mew=0.5, zorder=3)
    for q, d in zip(p, dx * NM * MAG):
        ax.annotate('', xy=(q[0] + d[0], q[1] + d[1]), xytext=tuple(q), zorder=5,
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color=C_STEP, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0.04, 0.97, f'steps $\\times${MAG}', transform=ax.transAxes, ha='left', va='top', fontsize=7)
    lab(ax, 'Motion model', 'c')
    arrow_between(fig, 2, '$\\Delta\\hat{\\mathbf{x}}_i^t$')

    ax = panel(3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    lab(ax, 'Excluded-volume\nmodule', 'd')
    j, i0 = np.array([8.2, 5.0]), np.array([1.0, 4.4])
    rr = 3.6
    ax.add_patch(Circle(j, rr, fill=False, ls='--', lw=0.7, ec='0.4'))
    ax.add_patch(Circle(j, 0.28, fc='white', ec='black', lw=0.7, zorder=4))
    ax.add_patch(Circle(i0, 0.28, fc='white', ec='black', lw=0.7, zorder=6))
    v = (j - i0) / np.linalg.norm(j - i0)
    t = np.array([-v[1], v[0]])
    rad, tang = 5.0 * v, 2.0 * t
    h = 0.65
    ex = tang + h * rad
    dash = dict(color='0.4', lw=0.6, ls=':', zorder=3)
    arr = lambda a, b, **k: ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', shrinkA=0, shrinkB=0, mutation_scale=5, **k), zorder=5)
    arr(i0, i0 + rad + tang, lw=0.9, color=C_STEP)
    ax.plot(*zip(i0, i0 + rad), **dash)
    ax.plot(*zip(i0, i0 + tang), **dash)
    ax.plot(*zip(i0 + rad, i0 + rad + tang), **dash)
    ax.plot(*zip(i0 + tang, i0 + rad + tang), **dash)
    print('d: red tip dist to j / r_core', round(np.linalg.norm(i0 + rad + tang - j) / rr, 2), 'black tip', round(np.linalg.norm(i0 + ex - j) / rr, 2))
    arr(i0, i0 + ex, lw=1.8, color='black')
    arr(i0, i0 + 1.6 * v, lw=0.6, color='0.4')
    ax.text(0.4, 9.75, 'proposed', fontsize=7, ha='left', va='top', color=C_STEP)
    ax.text(0.4, 8.3, 'taken', fontsize=7, ha='left', va='top', color='black')
    ax.plot(*zip(j, j + np.array([0, -rr])), color='0.4', lw=0.7, zorder=3)
    ax.text(j[0] - 0.4, j[1] - 1.9, 'hard\ncore', fontsize=7, ha='right', va='center', linespacing=1.0)
    arrow_between(fig, 3, '$\\hat{\\mathbf{x}}_i^{t+1}$')

    ax = panel(4)
    ax_e = ax
    ax.set_xlim(*ext)
    ax.set_ylim(*ext)
    tr = to_nm(xy[:TRACE, :, :]).reshape(TRACE, 7, 2)
    for k in range(7):
        s = tr[:, k]
        br = np.where(np.abs(np.diff(s, axis=0)).max(1) > L / 2)[0] + 1
        for seg in np.split(s, br):
            ax.plot(seg[:, 0], seg[:, 1], color='0.45', lw=0.5, zorder=2)
    p1 = to_nm(xy[1])
    ax.plot(p1[:, 0], p1[:, 1], 'o', ms=3.2, mfc='none', mec='black', mew=0.7, zorder=5)
    lab(ax, 'Next frame', 'e')
    sp_ax = [ax_a, ax_c, ax_e]
    assert all(a.get_xlim() == ext and a.get_ylim() == ext and a.get_aspect() == 1.0 for a in sp_ax) and ax_b.get_aspect() == 1.0
    assert len({tuple(np.round(a.get_position().bounds[2:], 6)) for a in sp_ax + [ax_b]}) == 1
    lines = lambda a: [l.get_xydata() for l in a.lines if l.get_marker() == 'o' and len(l.get_xdata()) == 7]
    pa, pc, pe = lines(ax_a)[0], lines(ax_c)[0], lines(ax_e)[-1]
    pb = [l.get_xydata() for l in ax_b.lines if l.get_marker() == 'o' and len(l.get_xdata()) == 4][0]
    db = pb - p[grp]
    db -= L * np.round(db / L)
    assert np.allclose(pa, p) and np.allclose(pc, p) and np.allclose(pe, p1) and np.abs(db).max() < 1e-6

    # loop back from e to b
    xb = X0 + 1 * (PW + GAP) + 0.05 / 2.54
    xe = X0 + 4 * (PW + GAP) + PW / 2
    yb, yl = Y0 - 0.02, 0.62 / 2.54
    xm = X0 + 1 * (PW + GAP) - GAP / 2
    xb, yt = X0 + 1 * (PW + GAP) - 0.02, Y0 + 0.35 / 2.54
    pts = [(xe, yb), (xe, yl), (xm, yl), (xm, yt)]
    fig.add_artist(plt.Line2D([p[0] / W for p in pts], [p[1] / H for p in pts], color='black', lw=0.7, transform=fig.transFigure))
    fig.add_artist(FancyArrowPatch((xm / W, yt / H), ((xb + 0.04) / W, yt / H), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=6, lw=0.7, color='black', shrinkA=0, shrinkB=0))
    xc = X0 + 2 * (PW + GAP) + PW / 2
    fig.add_artist(FancyArrowPatch((xc / W, yl / H), (xc / W, yb / H), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=6, lw=0.7, color='black', shrinkA=0, shrinkB=0))
    fig.text((xm + xe) / 2 / W, (yl - 0.04 / 2.54) / H,
             'positions of the last two frames, orientations, depths,\n motion latent and LSTM states; $t \\leftarrow t+1$',
             ha='center', va='top', fontsize=7)

    os.makedirs(OUT, exist_ok=True)
    for ext_, kw_ in (('.pdf', {}), ('.png', {})):
        fig.savefig(os.path.join(OUT, 'fig_overview' + ext_), dpi=600, **kw_)
    print('fig_overview figure size (cm)', round(fig.get_figwidth() * 2.54, 2), round(fig.get_figheight() * 2.54, 2), 'exact')
    plt.close(fig)


if __name__ == '__main__':
    main()
