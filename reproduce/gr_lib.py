"""g(r) score shared by the backfill and the comparison: pair-distance histograms on NBIN fine bins from 0 to box/sqrt(2)
(every STRIDE-th frame, lateral minimum image). The score of a set against an MD trajectory is KL(MD || set) of the histograms
merged by a factor f, divided by KL(MD || random placement), so 0 is MD and 1 is random. Bin width per system: the finest width at
which random placement sits at least twice as far from MD as the other MD trajectory does: 7-PIEZO f=1 (6.1 units), 14-PIEZO f=20 (161.5)."""
import json
import os
import sys

import numpy as np

from paths import ROOT

sys.path.insert(0, ROOT)
from membrane import fit_objective as fo
from membrane.FitObjectiveCallback import load_md

NBIN, STRIDE = 100, 4
MERGE = {7: 1, 14: 20}
CFG = {7: 'configs/7piezo/full.json', 14: 'configs/14piezo/full.json'}
GT = {7: ['data/membrane/half/7piezo_R1second.csv', 'data/membrane/half/7piezo_R2second.csv'],
      14: ['data/membrane/half/piezo14_rep01second.csv', 'data/membrane/half/piezo14_rep02second.csv']}


def edges(box):
    return np.linspace(0, box / np.sqrt(2), NBIN + 1)


def hist(x, box):
    """x: [..., T, P, >=2] positions; every STRIDE-th frame"""
    return np.histogram(fo.pair_dist_xy(np.asarray(x)[..., ::STRIDE, :, :2], box).ravel(), edges(box))[0].astype(np.float64)


def kl(hm, hx, f):
    hm, hx = hm.reshape(-1, f).sum(-1), np.asarray(hx, float).reshape(-1, f).sum(-1)
    p = hm / hm.sum()
    q = (hx + 0.5) / (hx + 0.5).sum()
    return float(np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0])))


class Ref:
    """MD histograms per trajectory and the random-placement histogram for one system"""
    def __init__(self, P):
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            self.box = float(json.load(open(CFG[P]))['box_size'])
            md = load_md(GT[P])
        finally:
            os.chdir(cwd)
        self.names = list(md)
        self.H = {k: hist(v, self.box) for k, v in md.items()}
        self.Hr = np.histogram(fo.pair_dist_xy(np.random.default_rng(0).random((20000, P, 2)) * self.box, self.box).ravel(),
                               edges(self.box))[0].astype(np.float64)
        self.f = MERGE[P]

    def score(self, k, hx, f=None):
        f = f or self.f
        return kl(self.H[k], hx, f) / kl(self.H[k], self.Hr, f)
