"""Side-by-side videos of MD (left) and the full model's best held-out rollout (right) on second-half trajectory 2, per system.
Best = the seed with the lowest held-out kappa on that trajectory, then its rollout with the lowest per-rollout kappa (same
H.kl_metrics call as kl_table). Rollout frame j is compared with MD frame j (index 0 is the shared start; node depth of the rollout
starts at frame 1, so frames run 1, 1+STRIDE, ...). Positions are wrapped into the box after one shift for the whole video, the same
for both panels: the circular mean of all proteins at the shared start (frame 0) goes to the box centre.
Hyperedge junctions are the periodic circular mean of the members' positions per axis (as in the membrane module).
Large circles are coloured by the depth under the protein, small dots (one per hyperedge, all of them every frame) by the depth at the
junction of the group, drawn at that junction. The model's chosen rollout, with its
hyperedge depth, is read from assets/videos/<system>_traj2_hyper.npz (roll_hyper_video.py, which re-runs the rollout batch of roll_heldout.py and
checks it against the saved one). Writes videos/{7,14}piezo_traj2_centered.mp4 and a still png each.
usage: video.py [7|14]"""
import os
import subprocess
import sys

import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection

from common import SEED
from figures import H, fo, plt, setup
from paths import HYPER, ROLLS, VIDEOS

OUT = VIDEOS
STRIDE, FPS = 5, 30
JOBS = {'7': ('7-PIEZO', '7piezo_R2second', '7piezo_traj2'), '14': ('14-PIEZO', 'piezo14_rep02second', '14piezo_traj2')}
W, HT, DPI = 12.8, 7.2, 100
SH = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]     # periodic images, so a circle on the edge shows on both sides


def pick(s, key):
    rows = s['rows']['sq']
    i = int(np.argmin([r['kl'][key]['logpair_kl_share'] for r in rows]))
    p, row = s['picks']['sq'][i], rows[i]
    assert p['tag'] == row['tag']
    z = np.load(os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz"))
    x, dep = z['xy_' + key], z['node_depth_' + key]
    md = s['md'][key]
    rnd = np.random.default_rng(0).random((8000, s['P'], 2)) * s['box']
    kall = H.kl_metrics(x, md, rnd, s['box'])['logpair_kl_share']
    assert abs(kall / row['kl'][key]['logpair_kl_share'] - 1) < 1e-2, (kall, row['kl'][key]['logpair_kl_share'])
    ks = [H.kl_metrics(x[j:j + 1], md, rnd, s['box'])['logpair_kl_share'] for j in range(len(x))]
    j = int(np.argmin(ks))
    other = next(k for k in s['md'] if k != key)
    floor = H.kl_metrics(s['md'][other][None], md, rnd, s['box'])['logpair_kl_share']
    print(f"{s['name']}: seed {row['seed']} ({p['tag']}) rollout {j} kappa {ks[j]:.4f}, seed kappa {row['kl'][key]['logpair_kl_share']:.4f} "
          f"(all 32 recomputed {kall:.4f}), MD floor {floor:.4f}", flush=True)
    return x[j], dep[j], p, j


def contacts(p, L, rc):
    """p [P, 2] nm wrapped -> segments of every pair within rc (minimum image), both halves for a pair across the boundary"""
    i, j = np.triu_indices(len(p), 1)
    d = p[j] - p[i]
    d -= L * np.round(d / L)
    m = np.hypot(d[:, 0], d[:, 1]) <= rc
    i, j, d = i[m], j[m], d[m]
    return [np.stack([p[a], p[a] + dd]) for a, dd in zip(i, d)] + [np.stack([p[b], p[b] - dd]) for b, dd in zip(j, d)]


def junctions(p, Hmean, L):
    """p [P, 2] wrapped -> [H, 2] periodic circular mean of each hyperedge's members per axis, wrapped into [0, L)"""
    th = 2 * np.pi * p / L
    return np.mod(np.arctan2(Hmean @ np.sin(th), Hmean @ np.cos(th)), 2 * np.pi) * L / (2 * np.pi)


def render(name, key, out):
    s = setup(name)
    s['name'] = name
    nm, L, P = s['nm'], s['box'] * s['nm'], s['P']
    z = np.load(os.path.join(HYPER, os.path.basename(out).replace('_centered.mp4', '_hyper.npz')))
    xr, dr, hr, mem = z['xy'], z['node'], z['hyper'], z['members'].astype(bool)
    print(f"{name}: rollout {int(z['rollout'])} of {z['tag']}", flush=True)
    xm, dm = s['md'][key][..., :2], s['depth'][key]['node']
    hm = s['depth'][key]['hyper']
    assert len(xm) == len(xr) and len(dm) == len(xm) and len(dr) == len(xr) - 1
    fr = np.arange(1, len(xm), STRIDE)        # depth of the rollout at frame j is dr[j - 1]; MD depth index equals its frame index
    a = 2 * np.pi * np.mod(xm[0] * nm, L) / L
    sh = L / 2 - np.mod(np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0)), 2 * np.pi) * L / (2 * np.pi)
    print(f'{name}: shift {sh[0]:.1f}, {sh[1]:.1f} nm; start frames differ by {np.abs(xm[0] - xr[0]).max() * nm:.4f} nm', flush=True)
    pm, pr = np.mod(xm[fr] * nm + sh, L), np.mod(xr[fr] * nm + sh, L)
    dmv, drv, hmv, hrv = dm[fr] * nm, dr[fr - 1] * nm, hm[fr] * nm, hr[fr - 1] * nm
    hi = np.nanpercentile(np.abs(np.concatenate([a.ravel() for a in (dmv, drv, hmv, hrv)])), 99)
    lo = -hi
    Hmean = mem / mem.sum(1, keepdims=True)
    print(f'{name}: depth limits +-{hi:.2f} nm, frames {len(fr)} ({fr[0]}..{fr[-1]}), core {s["core"] * nm:.2f} nm, rc {s["rc"] * nm:.2f} nm', flush=True)
    big = [fo.largest_cluster(fo.pair_dist_xy(x[fr], s['box']), s['rc']) for x in (xm, xr)]
    D = s['core'] * nm
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('0.6')
    fig = plt.figure(figsize=(W, HT), dpi=DPI)
    side, ys = 5.0 / W, 5.0 / HT
    arts = []
    for x0, ttl, pos, dep, hdep, bg in ((0.06, 'MD', pm, dmv, hmv, big[0]), (0.50, 'model', pr, drv, hrv, big[1])):
        ax = fig.add_axes([x0, 0.13, side, ys])
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=15, pad=6)
        con = LineCollection([], colors='0.45', linewidths=0.8, zorder=2)
        ax.add_collection(con)
        Hn, dd = len(mem), 0.22 * D
        hc = EllipseCollection(np.full(9 * Hn, dd), np.full(9 * Hn, dd), np.zeros(9 * Hn), units='xy', offsets=np.zeros((9 * Hn, 2)), offset_transform=ax.transData,
                               cmap=cmap, edgecolors='0.35', linewidths=0.3, zorder=2.5)
        hc.set_clim(lo, hi)
        ax.add_collection(hc)
        ec = EllipseCollection(np.full(9 * P, D), np.full(9 * P, D), np.zeros(9 * P), units='xy', offsets=np.zeros((9 * P, 2)), offset_transform=ax.transData,
                               cmap=cmap, edgecolors='0.15', linewidths=0.6, zorder=3)
        ec.set_clim(lo, hi)
        ax.add_collection(ec)
        txt = ax.text(0.5, -0.045, '', transform=ax.transAxes, ha='center', va='top', fontsize=13)
        arts.append((ec, con, hc, txt, pos, dep, hdep, bg))
    cax = fig.add_axes([0.925, 0.13, 0.014, ys])
    cb = fig.colorbar(arts[0][0], cax=cax)
    cb.set_label('depth under protein (nm)', fontsize=12)
    cb.ax.tick_params(labelsize=11)
    fig.suptitle(f'{name}, trajectory 2', fontsize=17, y=0.965)
    fig.text(0.5, 0.915, 'large circles: depth under protein; small dots: depth at group junctions', ha='center', va='center', fontsize=12)
    ft = fig.text(0.5, 0.03, '', ha='center', va='center', fontsize=13)

    def draw(f):
        for ec, con, hc, txt, pos, dep, hdep, bg in arts:
            p = pos[f]
            ec.set_offsets(np.concatenate([p + L * np.array(o) for o in SH]))
            ec.set_array(np.ma.masked_invalid(np.tile(dep[f], 9)))
            con.set_segments(contacts(p, L, s['rc'] * nm))
            cen = junctions(p, Hmean, L)
            hc.set_offsets(np.concatenate([cen + L * np.array(o) for o in SH]))
            hc.set_array(np.ma.masked_invalid(np.tile(hdep[f], 9)))
            txt.set_text(f'largest cluster: {bg[f]}')
        ft.set_text(f'frame {fr[f]}')
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    os.makedirs(OUT, exist_ok=True)
    mid = len(fr) // 2
    draw(mid)
    fig.savefig(os.path.join(OUT, os.path.basename(out)[:-4] + '_still.png'), dpi=DPI)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{int(W * DPI)}x{int(HT * DPI)}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
    pr_ = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(len(fr)):
        pr_.stdin.write(draw(f).tobytes())
        if f % 200 == 0:
            print(f'{name}: frame {f}/{len(fr)}', flush=True)
    pr_.stdin.close()
    assert pr_.wait() == 0
    print('wrote', out, flush=True)


if __name__ == '__main__':
    for k in sys.argv[1:] or JOBS:
        n, key, f = JOBS[k]
        render(n, key, os.path.join(OUT, f + '_centered.mp4'))
