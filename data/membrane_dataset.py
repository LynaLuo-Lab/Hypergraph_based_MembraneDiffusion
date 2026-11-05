"""Membrane-conditioned dataset: aligned (coords, node-depth, hyperedge-depth)
windows with on-the-fly joint S7 x D4 augmentation.

The membrane depths are D4-INVARIANT scalars (D4 acts on xy only, z untouched),
so D4 transforms only the coordinates; the S7 particle permutation remaps BOTH
the coords (gather) and the depths (node slots + hyperedge subset slots),
keeping coords and membrane mutually consistent by construction.

Source artifact: data/membrane/membrane_train.npz
  coords [N,21] (delta-std normalized, scaled box), node_depth [N,7],
  hyper_depth [N,120], members [120,7] bool, ksize [120].

Mirrors MolecularSlidingWindowDataset (shared-memory seq_length for curriculum,
samples_per_epoch budget). __getitem__ -> (coords[S,21], node[S,7], hyper[S,120]).
"""
from __future__ import annotations

import math
import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset

from data.paths import resolve

BOX = 856.196          # scaled-box (model units); matches positions.npy normalization


def _apply_translation_xy(coords, shift, box):
    """Rigid xy translation of a [S,P,3] window, re-wrapped into [0, box).

    ONE shift for the WHOLE window, never per frame: a shift that varied with time
    would manufacture displacement that never happened and corrupt every
    lag-resolved metric (MSD, VACF, cohesion) at once.

    Nothing else in the window moves. node_depth is indexed by PROTEIN and
    hyper_depth by HYPEREDGE, not by grid cell, so both ride along unchanged --
    the same reason D4 leaves them alone. theta and blade are orientations and a
    translation does not rotate anything.
    """
    out = coords.copy()
    out[..., :2] = np.remainder(out[..., :2] + shift, box)
    return out


class PermutationSampler:
    """Uniform draw from S_P, or from the product of within-species S_k.

    Never enumerates. The previous code built itertools.permutations(range(P)) at
    construction: 5,040 arrays at P=7, but 87 BILLION at P=14, so a 14-particle
    system could not have reached the first training step. A fresh
    np.random.permutation is uniform on S_P by construction and costs O(P).

    species_groups keeps the draw legal on MIXED systems. Permuting a piezo slot
    onto a trek slot is not a symmetry -- the particles are not interchangeable --
    so each group is shuffled independently and slots never cross groups.

    identity_only disables permutation entirely, for systems where it is not a
    symmetry at all (e.g. one protein per species).
    """

    def __init__(self, n_particles, species_groups=None, identity_only=False,
                 rng=None):
        self.P = int(n_particles)
        self.identity_only = bool(identity_only)
        self.rng = rng if rng is not None else np.random.default_rng()
        if species_groups:
            groups = [np.asarray(g, dtype=np.int64) for g in species_groups]
            seen = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
            if seen.size != self.P or len(np.unique(seen)) != self.P:
                raise ValueError(
                    f'species_groups must partition all {self.P} slots exactly once; '
                    f'got {seen.size} entries covering {len(np.unique(seen))} slots')
            self.groups = groups
        else:
            self.groups = [np.arange(self.P, dtype=np.int64)]

    def size(self):
        """Group order, for reporting. math.factorial overflows nothing in Python."""
        n = 1
        for g in self.groups:
            n *= math.factorial(len(g))
        return 1 if self.identity_only else n

    def __call__(self):
        perm = np.arange(self.P, dtype=np.int64)
        if self.identity_only:
            return perm
        for g in self.groups:
            perm[g] = self.rng.permutation(g)
        return perm


def _apply_d4_xy(coords, rot_k, flip, box=BOX):
    """D4 on the xy of a [S,P,3] coord block, around the box center; z untouched."""
    xy = coords[..., :2] - box * 0.5
    for _ in range(rot_k):                                    # 90deg CCW: (x,y)->(-y,x)
        xy = np.stack([-xy[..., 1], xy[..., 0]], axis=-1)
    if flip:                                                  # reflect across y-axis
        xy = np.stack([-xy[..., 0], xy[..., 1]], axis=-1)
    xy = (xy + box * 0.5) % box
    return np.concatenate([xy, coords[..., 2:3]], axis=-1)


def _apply_d4_vec(v, rot_k, flip):
    """D4 on a [...,3] VECTOR (an offset, not a position): rotate/reflect xy, leave z.

    No box centring and no modulo -- those belong to _apply_d4_xy, which moves
    absolute positions. Applying them here would shift every blade by half a box.
    Correct under the reflection too, which is why blades are stored rather than an
    SO(3) frame (D*R leaves SO(3) when det(D) = -1)."""
    xy = v[..., :2]
    for _ in range(rot_k):
        xy = np.stack([-xy[..., 1], xy[..., 0]], axis=-1)
    if flip:
        xy = np.stack([-xy[..., 0], xy[..., 1]], axis=-1)
    return np.concatenate([xy, v[..., 2:3]], axis=-1)


def build_hyperedge_perm_table(members):
    """members [120,7] bool -> (slot_of, member_idx_list).

    slot_of: dict frozenset(original protein indices) -> canonical slot.
    Used to remap hyperedge depths under a particle permutation.
    """
    member_idx = [tuple(np.where(members[s])[0].tolist()) for s in range(members.shape[0])]
    slot_of = {frozenset(m): s for s, m in enumerate(member_idx)}
    return slot_of, member_idx


def hyperedge_remap(perm, slot_of, member_idx):
    """For particle permutation `perm` (augmented slot j holds original perm[j]),
    return remap[120] s.t. hyper_aug = hyper[remap]."""
    remap = np.empty(len(member_idx), dtype=np.int64)
    for s, m in enumerate(member_idx):
        remap[s] = slot_of[frozenset(perm[j] for j in m)]
    return remap


def _augment_window(coords, node, hyper, perm_fn, slot_of, member_idx, theta=None,
                    blade=None, translate=False, rng=None, box=BOX):
    """Joint S7 x D4 augmentation of one window. Depths are D4-invariant (D4 acts
    on xy only); the S7 permutation remaps coords and depth slots together so they
    stay mutually consistent. Shared by the single- and multi-trajectory datasets.

    When `theta` [S,7] (per-protein trimer orientation) is given, it is transformed
    by the SAME group action: S7 permute (theta follows its protein), D4 rotation
    adds 90*k rad-equivalent (+pi/2 per rot), and the y-axis reflection maps
    theta -> pi - theta. Matches _apply_d4_xy (rotate then flip).

    When `blade` [S,7,3,3] (per-protein blade offset vectors) is also given, it is
    permuted the same way as theta and transformed via _apply_d4_vec -- the SAME
    group element as theta, so the two stay mutually consistent."""
    perm = perm_fn() if callable(perm_fn) else perm_fn[
        int(torch.randint(0, len(perm_fn), (1,)).item())]
    rot_k = int(torch.randint(0, 4, (1,)).item())
    flip = int(torch.randint(0, 2, (1,)).item())
    coords = coords[:, perm, :]                              # augmented slot j = original perm[j]
    node = node[:, perm]
    hyper = hyper[:, hyperedge_remap(perm, slot_of, member_idx)]
    coords = _apply_d4_xy(coords, rot_k, flip, box)          # D4 on xy only
    if translate:
        # After D4, not before: D4 acts about the box centre, so translating first
        # would put the window somewhere else entirely and the two would not compose
        # into the intended group element. Continuous, so it is the diversity source
        # that does not depend on P -- unlike S_P, whose size runs from 2 to 14!.
        r = rng if rng is not None else np.random
        coords = _apply_translation_xy(coords, r.uniform(0.0, box, size=2), box)
    if theta is not None:
        th = theta[:, perm] + rot_k * (np.pi / 2.0)          # S7 + D4 rotation (90deg CCW/rot)
        if flip:
            th = np.pi - th                                  # reflect across y-axis
        if blade is not None:
            bl = _apply_d4_vec(blade[:, perm], rot_k, flip)  # same group element
            return coords, node, hyper, th, bl
        return coords, node, hyper, th
    return coords, node, hyper


def check_membrane_system(ds, config):
    """The config's system has to be the data's: coordinate width 3 * n_particles, a members table over n_particles
    proteins, and the box the coords were wrapped in. A mismatch otherwise trains on folded or misread windows, or fails
    deep in the model with a shape error."""
    n = config.get('n_particles', 7)
    coords = ds.coords if isinstance(ds.coords, list) else [ds.coords]
    widths = sorted({c.shape[1] for c in coords})
    if widths != [3 * n] or ds.members.shape[1] != n:
        raise ValueError(f'membrane data has coordinate width {widths} and members over {ds.members.shape[1]} '
                         f'proteins, but n_particles={n}')
    box = config.get('box_size', BOX)
    if abs(ds.box - box) > 1e-3:
        raise ValueError(f'membrane data was wrapped in box {ds.box}, config box_size is {box}')


class MembraneWindowDataset(Dataset):
    """Random-start windows of (coords, node-depth, hyperedge-depth) with joint
    S7 x D4 augmentation applied per window."""

    def __init__(self, npz_path='data/membrane/membrane_train.npz', seq_length=50,
                 samples_per_epoch=128, augment=True, n_particles=7, seed=0,
                 return_theta=False, return_blade=False,
                 augment_translate=True, augment_permute=True, species_groups=None):
        d = np.load(resolve(npz_path), allow_pickle=True)
        # the box the coords were wrapped in: a system other than 7-PIEZO stores it, 7-PIEZO npz files do not
        self.box = float(d['box_size']) if 'box_size' in d else BOX
        self.return_theta = bool(return_theta)
        if self.return_theta:
            if 'theta' not in d:
                raise ValueError(f'{npz_path} has no "theta" (run scripts/membrane/data_prep/build_membrane_arms.py)')
            self.theta = d['theta'].astype(np.float32)          # [N,7] per-protein orientation
            self.arm_radius = float(d['arm_radius'])
        self.return_blade = bool(return_blade)
        if self.return_blade and not self.return_theta:
            raise ValueError('return_blade requires return_theta=True: blade is only '
                             'carried through augmentation alongside theta, which supplies '
                             'the group element it must move under')
        if self.return_blade:
            if 'blade_vec' not in d:
                raise ValueError(f'{npz_path} has no "blade_vec" '
                                 f'(rerun scripts/membrane/data_prep/build_membrane_arms.py)')
            self.blade = d['blade_vec'].astype(np.float32)   # [N,7,3,3]
        self.coords = d['coords'].astype(np.float32)            # [N,21]
        self.node = d['node_depth'].astype(np.float32)          # [N,7]
        self.hyper = d['hyper_depth'].astype(np.float32)        # [N,120]
        self.members = d['members'].astype(bool)                # [120,7]
        # Normalize depths to ~unit variance. The npz stores /delta_std depths at
        # |mean|~65 (node std ~28), so the raw-MSE aux loss was on a scale ~1e3
        # while w_membrane=10 sat against physics weights in the thousands -> the
        # aux term was negligible and the membrane module COLLAPSED to ~5% of GT
        # magnitude. Unit-scaling puts the aux MSE on par with the other losses so
        # a reasonable w_membrane can actually train the membrane. Stds saved for
        # de-normalization at inference/visualization.
        self.node_std = float(self.node.std()) or 1.0
        self.hyper_std = float(self.hyper.std()) or 1.0
        self.node = self.node / self.node_std
        self.hyper = self.hyper / self.hyper_std
        self.P = int(n_particles)
        self.N = self.coords.shape[0]
        self.augment = bool(augment)
        self._shared_seq_length = multiprocessing.Value('i', int(seq_length))
        self.samples_per_epoch = int(samples_per_epoch)
        self._slot_of, self._member_idx = build_hyperedge_perm_table(self.members)
        # full S7 permutation list (5040) for N=7; D4 = 4 rot x 2 flip
        self._perm_sampler = PermutationSampler(
            self.P, species_groups=species_groups,
            identity_only=not bool(augment_permute),
            rng=np.random.default_rng(seed))
        self._translate = bool(augment_translate)
        self._aug_rng = np.random.default_rng(seed + 1)
        if self.augment:
            print(f'  membrane augmentation: D4=8 x perm={self._perm_sampler.size()}'
                  f' x translate={"continuous" if self._translate else "off"}',
                  flush=True)
        self._validate(seq_length)

    # --- curriculum shared seq_length ---
    @property
    def seq_length(self):
        with self._shared_seq_length.get_lock():
            return self._shared_seq_length.value

    @seq_length.setter
    def seq_length(self, value):
        with self._shared_seq_length.get_lock():
            self._shared_seq_length.value = int(value)

    def update_n_samples(self, current_len):
        self.seq_length = int(current_len)
        self._validate(current_len)

    def _validate(self, seq_len):
        if self.N < seq_len:
            raise ValueError(f'trajectory length {self.N} < seq_length {seq_len}')

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        S = self.seq_length
        # torch.randint draws from the per-worker-seeded RNG (Lightning seeds
        # workers), so multiple DataLoader workers don't draw identical windows.
        start = int(torch.randint(0, self.N - S + 1, (1,)).item())
        sl = slice(start, start + S)
        coords = self.coords[sl].reshape(S, self.P, 3).copy()   # [S,7,3]
        node = self.node[sl].copy()                             # [S,7]
        hyper = self.hyper[sl].copy()                           # [S,120]
        theta = self.theta[sl].copy() if self.return_theta else None   # [S,7]
        blade = self.blade[sl].copy() if self.return_blade else None   # [S,7,3,3]

        if self.augment:
            if self.return_theta:
                out_aug = _augment_window(
                    coords, node, hyper, self._perm_sampler, self._slot_of,
                    self._member_idx, theta=theta, blade=blade,
                    translate=self._translate, rng=self._aug_rng, box=self.box)
                if self.return_blade:
                    coords, node, hyper, theta, blade = out_aug
                else:
                    coords, node, hyper, theta = out_aug
            else:
                coords, node, hyper = _augment_window(
                    coords, node, hyper, self._perm_sampler, self._slot_of,
                    self._member_idx, translate=self._translate, rng=self._aug_rng, box=self.box)

        coords = coords.reshape(S, self.P * 3)
        out = [torch.from_numpy(coords).float(),
               torch.from_numpy(node).float(),
               torch.from_numpy(hyper).float()]
        if self.return_theta:
            out.append(torch.from_numpy(np.ascontiguousarray(theta)).float())
        if self.return_blade:
            out.append(torch.from_numpy(np.ascontiguousarray(blade)).float())
        return tuple(out)


class MultiMembraneWindowDataset(Dataset):
    """Pool of membrane trajectories: every window is drawn from one of several
    npz files, so a batch mixes initial conditions from all of them. Built for
    training the membrane fork on both 7piezo replicas (traj1 + traj2) at once,
    so the model sees the ensemble spread instead of overfitting one trajectory.

    Two things must be shared across the pool, and both are enforced:
      - the hyperedge structure (`members`), since the S7 remap depends on it;
      - the depth normalization std -- one std over ALL trajectories, so the
        membrane signal is on the same scale for every replica (the single-traj
        dataset normalizes by its own std, which would put two replicas on
        different scales if reused blindly).

    `return_blade` pools `blade_vec` the same way `return_theta` pools `theta`:
    one list entry per npz, collected alongside theta and passed through
    `_augment_window` under the same draw. `return_blade=True` requires
    `return_theta=True` -- blade only moves consistently under augmentation
    when theta supplies the group element (perm/rot_k/flip) it rides along with.

    Sampling (`balance`):
      - False (default): flat union of window-starts -- each valid IC across the
        whole pool is equally likely, so a longer trajectory contributes
        proportionally more windows. This is the literal "each batch could be an
        IC from either dataset".
      - True: pick a trajectory uniformly first, then a window -- equal weight per
        replica regardless of length.
    """

    def __init__(self, npz_paths, seq_length=50, samples_per_epoch=128,
                 augment=True, n_particles=7, seed=0, balance=False,
                 return_theta=False, return_blade=False,
                 augment_translate=True, augment_permute=True, species_groups=None):
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]
        if not npz_paths:
            raise ValueError('npz_paths is empty')
        self.P = int(n_particles)
        self.augment = bool(augment)
        self.balance = bool(balance)
        self.samples_per_epoch = int(samples_per_epoch)
        self.paths = [resolve(p) for p in npz_paths]
        self.return_theta = bool(return_theta)
        self.return_blade = bool(return_blade)
        if self.return_blade and not self.return_theta:
            raise ValueError('return_blade requires return_theta=True: blade is only '
                             'carried through augmentation alongside theta, which supplies '
                             'the group element it must move under')

        coords_l, node_l, hyper_l, members0 = [], [], [], None
        theta_l, radii, blade_l, boxes = [], [], [], []
        for p in self.paths:
            d = np.load(p, allow_pickle=True)
            boxes.append(float(d['box_size']) if 'box_size' in d else BOX)
            coords_l.append(d['coords'].astype(np.float32))
            node_l.append(d['node_depth'].astype(np.float32))
            hyper_l.append(d['hyper_depth'].astype(np.float32))
            if self.return_theta:
                if 'theta' not in d:
                    raise ValueError(f'{p} has no "theta" (run scripts/membrane/data_prep/build_membrane_arms.py)')
                theta_l.append(d['theta'].astype(np.float32))
                radii.append(float(d['arm_radius']))
            if self.return_blade:
                if 'blade_vec' not in d:
                    raise ValueError(f'{p} has no "blade_vec" (rerun scripts/membrane/data_prep/build_membrane_arms.py)')
                blade_l.append(d['blade_vec'].astype(np.float32))
            m = d['members'].astype(bool)
            if members0 is None:
                members0 = m
            elif not np.array_equal(members0, m):
                raise ValueError(f'{p}: members table differs from the first npz -- '
                                 'hyperedge structure must match across the pool')
        self.members = members0
        if max(boxes) - min(boxes) > 1e-6:
            raise ValueError(f'box differs across the pool {boxes}: one box cannot wrap them all')
        self.box = boxes[0]

        # One arm_radius for the pool: the trimer radius is a property of the
        # protein, so replicas must agree (they come out ~0.2% apart). Averaging a
        # real disagreement would silently deform the reconstructed blades, so the
        # spread is checked rather than assumed.
        if self.return_theta:
            r = np.array(radii)
            spread = float(np.ptp(r) / r.mean())
            if spread > 0.02:
                raise ValueError(f'arm_radius differs by {100 * spread:.1f}% across the '
                                 f'pool {radii} -- one radius cannot describe both')
            self.arm_radius = float(r.mean())
            self.theta = theta_l
        if self.return_blade:
            self.blade = blade_l

        # SHARED normalization: one std across the whole pool (not per-file).
        self.node_std = float(np.concatenate([n.ravel() for n in node_l]).std()) or 1.0
        self.hyper_std = float(np.concatenate([h.ravel() for h in hyper_l]).std()) or 1.0
        self.coords = coords_l
        self.node = [n / self.node_std for n in node_l]
        self.hyper = [h / self.hyper_std for h in hyper_l]
        self.Ns = np.array([c.shape[0] for c in self.coords], dtype=np.int64)
        self.n_traj = len(self.coords)
        self.N = int(self.Ns.sum())

        self._shared_seq_length = multiprocessing.Value('i', int(seq_length))
        self._slot_of, self._member_idx = build_hyperedge_perm_table(self.members)
        self._perm_sampler = PermutationSampler(
            self.P, species_groups=species_groups,
            identity_only=not bool(augment_permute),
            rng=np.random.default_rng(seed))
        self._translate = bool(augment_translate)
        self._aug_rng = np.random.default_rng(seed + 1)
        if self.augment:
            print(f'  membrane augmentation: D4=8 x perm={self._perm_sampler.size()}'
                  f' x translate={"continuous" if self._translate else "off"}',
                  flush=True)
        self._validate(seq_length)

    @property
    def seq_length(self):
        with self._shared_seq_length.get_lock():
            return self._shared_seq_length.value

    @seq_length.setter
    def seq_length(self, value):
        with self._shared_seq_length.get_lock():
            self._shared_seq_length.value = int(value)

    def update_n_samples(self, current_len):
        self.seq_length = int(current_len)
        self._validate(current_len)

    def _validate(self, seq_len):
        if self.Ns.min() < seq_len:
            raise ValueError(f'shortest pooled trajectory {int(self.Ns.min())} '
                             f'< seq_length {seq_len}')

    def __len__(self):
        return self.samples_per_epoch

    def _choose_traj(self, S):
        avail = self.Ns - S + 1                              # valid window-starts per traj
        if self.balance:
            return int(torch.randint(0, self.n_traj, (1,)).item())
        w = torch.from_numpy((avail / avail.sum()).astype(np.float64))
        return int(torch.multinomial(w, 1).item())

    def __getitem__(self, idx):
        S = self.seq_length
        t = self._choose_traj(S)
        start = int(torch.randint(0, int(self.Ns[t]) - S + 1, (1,)).item())
        sl = slice(start, start + S)
        coords = self.coords[t][sl].reshape(S, self.P, 3).copy()
        node = self.node[t][sl].copy()
        hyper = self.hyper[t][sl].copy()
        theta = self.theta[t][sl].copy() if self.return_theta else None   # [S,7]
        blade = self.blade[t][sl].copy() if self.return_blade else None   # [S,7,3,3]

        if self.augment:
            if self.return_theta:
                out_aug = _augment_window(
                    coords, node, hyper, self._perm_sampler, self._slot_of,
                    self._member_idx, theta=theta, blade=blade,
                    translate=self._translate, rng=self._aug_rng, box=self.box)
                if self.return_blade:
                    coords, node, hyper, theta, blade = out_aug
                else:
                    coords, node, hyper, theta = out_aug
            else:
                coords, node, hyper = _augment_window(
                    coords, node, hyper, self._perm_sampler, self._slot_of,
                    self._member_idx, translate=self._translate, rng=self._aug_rng, box=self.box)

        coords = coords.reshape(S, self.P * 3)
        out = [torch.from_numpy(coords).float(),
               torch.from_numpy(node).float(),
               torch.from_numpy(hyper).float()]
        if self.return_theta:
            out.append(torch.from_numpy(np.ascontiguousarray(theta)).float())
        if self.return_blade:
            out.append(torch.from_numpy(np.ascontiguousarray(blade)).float())
        return tuple(out)


def membrane_collate(batch):
    coords = torch.stack([b[0] for b in batch])                 # [B,S,21]
    node = torch.stack([b[1] for b in batch])                   # [B,S,7]
    hyper = torch.stack([b[2] for b in batch])                  # [B,S,120]
    n = len(batch[0])
    if n > 3:                                                   # arms: per-protein theta
        theta = torch.stack([b[3] for b in batch])              # [B,S,7]
        if n > 4:                                               # blade: per-protein offset vectors
            blade = torch.stack([b[4] for b in batch])          # [B,S,7,3,3]
            return coords, node, hyper, theta, blade
        return coords, node, hyper, theta
    return coords, node, hyper


# --------------------------------------------------------------------------
#  self-test: augmentation consistency
# --------------------------------------------------------------------------
if __name__ == '__main__':
    ds = MembraneWindowDataset(seq_length=8, samples_per_epoch=4, augment=False)
    print(f'N={ds.N} coords{ds.coords.shape} node{ds.node.shape} hyper{ds.hyper.shape}')

    # (1) hyperedge remap is a bijection for a random perm
    rng = np.random.default_rng(1)
    perm = rng.permutation(7)
    remap = hyperedge_remap(perm, ds._slot_of, ds._member_idx)
    assert sorted(remap.tolist()) == list(range(120)), 'remap not a bijection'
    print('OK remap bijection')

    # (2) k=2 hyperedge remap matches pairwise relabeling
    #     pair {i,j} -> {perm[i],perm[j]}; check a few
    pair_slots = [s for s, m in enumerate(ds._member_idx) if len(m) == 2]
    for s in pair_slots[:5]:
        i, j = ds._member_idx[s]
        want = ds._slot_of[frozenset((perm[i], perm[j]))]
        assert remap[s] == want
    print('OK k=2 remap matches pair relabeling')

    # (3) perm then inverse-perm returns original depths
    inv = np.argsort(perm)
    n0 = ds.node[:5]
    n1 = n0[:, perm][:, inv]
    assert np.allclose(n0, n1)
    h0 = ds.hyper[:5]
    remap_inv = hyperedge_remap(inv, ds._slot_of, ds._member_idx)
    h1 = h0[:, remap][:, remap_inv]
    assert np.allclose(h0, h1)
    print('OK perp/inverse round-trip (node + hyper)')

    # (4) D4 leaves depths untouched, transforms coords; identity D4 is a no-op
    c = ds.coords[:3].reshape(3, 7, 3).copy()
    c_id = _apply_d4_xy(c, 0, 0)
    assert np.allclose(c % ds.coords.max(), c_id % ds.coords.max(), atol=1e-3) or np.allclose(c, c_id, atol=1e-3)
    c_rot = _apply_d4_xy(c, 1, 0)
    assert not np.allclose(c[..., :2], c_rot[..., :2]) and np.allclose(c[..., 2], c_rot[..., 2])
    print('OK D4 rotates xy, leaves z')

    # (5) augmented __getitem__ shapes
    ds.augment = True
    cc, nn, hh = ds[0]
    assert cc.shape == (8, 21) and nn.shape == (8, 7) and hh.shape == (8, 120)
    b = membrane_collate([ds[0], ds[1]])
    assert b[0].shape == (2, 8, 21) and b[1].shape == (2, 8, 7) and b[2].shape == (2, 8, 120)
    print('OK getitem + collate shapes')
    print('ALL DATASET SELF-TESTS PASSED')
