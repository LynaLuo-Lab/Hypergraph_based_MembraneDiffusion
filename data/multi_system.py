"""Multi-system variable-occupancy dataset.

Handles systems with different particle counts (1, 2, 7, 14 …) by padding
everything to MAX_PARTICLES=14 slots.  Each __getitem__ draw:

  1. Picks a system uniformly at random (balanced, regardless of view count).
  2. Picks a random view and random start within that view.
  3. Converts raw coords to fractional (÷ box_size, mod 1.0).
  4. Randomly permutes the n active particles into n of the 14 slots.
  5. Returns (window, active_mask, edge_src, edge_dst, edge_mask).

Constants
---------
MAX_PARTICLES      = 14   (hard slot capacity)
MAX_DIRECTED_EDGES = 182  = 14 * 13 (fully-connected directed graph capacity)
"""
from __future__ import annotations

import multiprocessing
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_PARTICLES: int = 14
MAX_DIRECTED_EDGES: int = MAX_PARTICLES * (MAX_PARTICLES - 1)  # 182


# ---------------------------------------------------------------------------
# Edge builder
# ---------------------------------------------------------------------------

def build_edges_for_active(
    active_mask: Tensor,
    max_edges: int = MAX_DIRECTED_EDGES,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build padded directed edge index tensors for active particles.

    Parameters
    ----------
    active_mask : BoolTensor [MAX_PARTICLES]
        True for slots that hold a real particle.
    max_edges : int
        Capacity to pad to (must be >= n_active*(n_active-1)).

    Returns
    -------
    edge_src  : LongTensor [max_edges]
    edge_dst  : LongTensor [max_edges]
    edge_mask : BoolTensor [max_edges]  — True for real edges, False for padding
    """
    active_slots = active_mask.nonzero(as_tuple=True)[0]  # [n_active]
    n_active = active_slots.numel()

    edge_src = torch.zeros(max_edges, dtype=torch.long)
    edge_dst = torch.zeros(max_edges, dtype=torch.long)
    edge_mask = torch.zeros(max_edges, dtype=torch.bool)

    if n_active < 2:
        return edge_src, edge_dst, edge_mask

    k = 0
    slots = active_slots.tolist()
    for i in slots:
        for j in slots:
            if i != j:
                edge_src[k] = i
                edge_dst[k] = j
                edge_mask[k] = True
                k += 1

    return edge_src, edge_dst, edge_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiSystemDataset(Dataset):
    """Variable-occupancy dataset across multiple particle systems.

    Parameters
    ----------
    system_configs : list of dict, each with keys:
        - name        : str
        - n_particles : int
        - box_size    : float  (assumed cubic; coords divided by this)
        - array       : np.ndarray of shape [n_views, T, n_particles*3]
    seq_length : int
        Initial window length (changeable via shared-memory property).
    samples_per_epoch : int
        How many __getitem__ draws constitute one epoch.
    """

    def __init__(
        self,
        system_configs: List[dict],
        seq_length: int = 50,
        samples_per_epoch: int = 128,
    ):
        if not system_configs:
            raise ValueError("MultiSystemDataset needs at least one system config.")

        self.systems = list(system_configs)
        self._shared_seq_length = multiprocessing.Value('i', int(seq_length))
        self.samples_per_epoch = int(samples_per_epoch)

        # Pre-validate
        for cfg in self.systems:
            arr = cfg["array"]
            n_p = cfg["n_particles"]
            expected_D = n_p * 3
            if arr.ndim != 3:
                raise ValueError(
                    f"System '{cfg['name']}' array must be 3D [n_views, T, D]; "
                    f"got ndim={arr.ndim}"
                )
            if arr.shape[2] != expected_D:
                raise ValueError(
                    f"System '{cfg['name']}' array D={arr.shape[2]} but "
                    f"n_particles={n_p} implies D={expected_D}"
                )
            if n_p > MAX_PARTICLES:
                raise ValueError(
                    f"System '{cfg['name']}' n_particles={n_p} exceeds "
                    f"MAX_PARTICLES={MAX_PARTICLES}"
                )

        self._validate(seq_length)

    # --- seq_length shared-memory accessors (curriculum callback compatible) ---

    @property
    def seq_length(self) -> int:
        with self._shared_seq_length.get_lock():
            return self._shared_seq_length.value

    @seq_length.setter
    def seq_length(self, value: int):
        with self._shared_seq_length.get_lock():
            self._shared_seq_length.value = int(value)

    def update_n_samples(self, current_len: int):
        """Called by the curriculum callback to update active seq length."""
        self.seq_length = int(current_len)
        self._validate(current_len)

    def _validate(self, seq_len: int):
        for cfg in self.systems:
            T = cfg["array"].shape[1]
            if T < seq_len:
                raise ValueError(
                    f"System '{cfg['name']}' trajectory length T={T} < seq_length={seq_len}"
                )

    # --- Dataset protocol ---

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        current_len = self.seq_length

        # 1. Balanced system selection (equal weight per system)
        n_systems = len(self.systems)
        sys_idx = int(torch.randint(0, n_systems, (1,)).item())
        cfg = self.systems[sys_idx]

        arr = cfg["array"]           # [n_views, T, D]
        n_views = arr.shape[0]
        T = arr.shape[1]
        n_particles = cfg["n_particles"]
        box_size = float(cfg["box_size"])

        # 2. Random view and start
        view_idx = int(torch.randint(0, n_views, (1,)).item())
        max_start = T - current_len
        start = int(torch.randint(0, max_start + 1, (1,)).item())

        raw = arr[view_idx, start:start + current_len]  # [T, n_particles*3]
        # Copy out of mmap-backed read-only slice
        raw = np.ascontiguousarray(raw).copy()
        window_np = raw.reshape(current_len, n_particles, 3)  # [T, n, 3]

        # 3. Fractional coords: divide by box_size, mod 1.0
        window_frac = (window_np / box_size) % 1.0  # [T, n, 3]

        # 4. Random slot assignment: pick n_particles out of MAX_PARTICLES slots
        perm = torch.randperm(MAX_PARTICLES)
        assigned_slots = perm[:n_particles]  # which slots get particles

        # Build output window [T, MAX_PARTICLES * 3], zeroed
        out_window = torch.zeros(current_len, MAX_PARTICLES * 3, dtype=torch.float32)
        # Build active_mask [MAX_PARTICLES]
        active_mask = torch.zeros(MAX_PARTICLES, dtype=torch.bool)

        for local_i, slot in enumerate(assigned_slots.tolist()):
            # Place particle local_i into slot
            slot_start = slot * 3
            out_window[:, slot_start:slot_start + 3] = torch.from_numpy(
                window_frac[:, local_i, :].astype(np.float32)
            )
            active_mask[slot] = True

        # 5. Build edge indices for active slots
        edge_src, edge_dst, edge_mask = build_edges_for_active(active_mask, MAX_DIRECTED_EDGES)

        return out_window, active_mask, edge_src, edge_dst, edge_mask


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def multi_system_collate(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Stack a list of (window, active_mask, edge_src, edge_dst, edge_mask) tuples.

    Returns
    -------
    window_batch    : [B, T, MAX_PARTICLES*3]
    active_mask_b   : [B, MAX_PARTICLES]
    edge_src_batch  : [B, MAX_DIRECTED_EDGES]
    edge_dst_batch  : [B, MAX_DIRECTED_EDGES]
    edge_mask_batch : [B, MAX_DIRECTED_EDGES]
    """
    windows, active_masks, edge_srcs, edge_dsts, edge_masks = zip(*batch)
    return (
        torch.stack(windows),
        torch.stack(active_masks),
        torch.stack(edge_srcs),
        torch.stack(edge_dsts),
        torch.stack(edge_masks),
    )
