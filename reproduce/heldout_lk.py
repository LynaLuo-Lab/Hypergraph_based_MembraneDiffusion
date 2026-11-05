"""Log pair-distance scores of a set of rollouts against MD: the Gaussian KL of ln pair distance with its signed mean and spread
parts, and the same on the change in ln pair distance over lags 1, 10, 100, 1000, each as a share of random placement's value."""
import sys

import numpy as np

from paths import ROOT

sys.path.insert(0, ROOT)
from membrane import fit_objective as fo

STRIDE, DLAGS = 4, (1, 10, 100, 1000)


def gparts(t, p):
    """KL(N_t || N_p) and its signed mean and spread parts (+ prediction larger / wider)"""
    mt, st, mp, sp = t.mean(), t.std(), p.mean(), p.std()
    mean = (mt - mp) ** 2 / (2 * sp ** 2)
    spread = np.log(sp / st) + st ** 2 / (2 * sp ** 2) - 0.5
    return mean + spread, np.sign(mp - mt) * mean, np.sign(sp - st) * spread


def lnpd(x, box):
    return np.log(np.maximum(fo.pair_dist_xy(x, box), 1.0))


def kl_metrics(x, md, rnd, box):
    """x [N, T, P, >=2] rollouts, md [T, P, >=2] MD, rnd [M, P, 2] random: log-distance KL and change KL as shares"""
    lx, lm, lr = lnpd(x[:, ::STRIDE], box), lnpd(md[::STRIDE], box), lnpd(rnd, box)
    k, m, s = gparts(lm.ravel(), lx.ravel())
    kr = gparts(lm.ravel(), lr.ravel())[0]
    lxf, lmf = lnpd(x, box), lnpd(md, box)
    d = [gparts((lmf[L:] - lmf[:-L]).ravel(), (lxf[:, L:] - lxf[:, :-L]).ravel()) for L in DLAGS]
    return {'logpair_kl_share': k / kr, 'logpair_mean_share': m / kr, 'logpair_spread_share': s / kr,
            'dlogpair_kl': float(np.mean([v[0] for v in d])), 'dlogpair_mean': float(np.mean([v[1] for v in d])),
            'dlogpair_spread': float(np.mean([v[2] for v in d]))}
