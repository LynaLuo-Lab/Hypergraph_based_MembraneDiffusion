"""Datasets.

Two datasets are exposed:

  - MolecularSlidingWindowDataset: single [T, D] torch-tensor trajectory.
    Random-start sampling, len() = samples_per_epoch.

  - MultiTrajectoryDataset: one or more [N_i, T, D] numpy arrays (typically
    memory-mapped augmented trajectory files). Random selection across the
    pooled set of views, then random start within the chosen view.

Both expose a shared-memory seq_length (multiprocessing.Value) so the
curriculum-learning callback can change the window length mid-epoch without
restarting DataLoader workers.

Epoch budget is decoupled from trajectory length: samples_per_epoch is a
configuration knob, not a function of (T, seq_len). This is essential at
seq_len ~= tau_c where stride-1 sliding would produce thousands of highly
correlated windows per trajectory.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.paths import resolve


# KEEP: variable-length batches across curriculum changes
def safe_collate(batch):
    return torch.stack(batch)


# =========================================================================
#  Single-trajectory random-start dataset
# =========================================================================

class MolecularSlidingWindowDataset(Dataset):
    """Random-start windows from a single [T, D] tensor."""

    def __init__(self, data_tensor, seq_length=50, samples_per_epoch=128):
        if not torch.is_tensor(data_tensor):
            data_tensor = torch.as_tensor(data_tensor, dtype=torch.float32)
        self.data = data_tensor
        self._shared_seq_length = multiprocessing.Value('i', int(seq_length))
        self.samples_per_epoch = int(samples_per_epoch)
        self._validate(seq_length)

    # --- seq_length shared-memory accessors ---

    @property
    def seq_length(self):
        with self._shared_seq_length.get_lock():
            return self._shared_seq_length.value

    @seq_length.setter
    def seq_length(self, value):
        with self._shared_seq_length.get_lock():
            self._shared_seq_length.value = int(value)

    def update_n_samples(self, current_len):
        """Called by the curriculum callback to update the active seq length."""
        self.seq_length = int(current_len)
        self._validate(current_len)

    def _validate(self, seq_len):
        if len(self.data) < seq_len:
            raise ValueError(
                f"Trajectory length {len(self.data)} is shorter than seq_length {seq_len}"
            )

    # --- Dataset protocol ---

    def __len__(self):
        # Decouples 'one epoch' from trajectory length; set via config.
        return self.samples_per_epoch

    def __getitem__(self, idx):
        current_len = self.seq_length
        max_start = len(self.data) - current_len
        # idx is ignored; samples_per_epoch is a budget, not an index space.
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        return self.data[start:start + current_len].clone()


# =========================================================================
#  Multi-trajectory random-view + random-start dataset
# =========================================================================

class MultiTrajectoryDataset(Dataset):
    """Pooled random sampling across one or more [N_i, T, D] trajectory arrays.

    Each array may be a numpy memmap (mmap_mode='r') — we hold the ndarray
    references and only materialize the selected window as a torch tensor in
    __getitem__, so RAM usage stays proportional to (num_workers * batch_size
    * seq_len * D) instead of total array size.

    Per draw:
      - Global view index ~ Uniform[0, sum_i N_i) (each view equally likely).
      - Start offset ~ Uniform[0, T_i - seq_len] where T_i is the length of
        the chosen view's source trajectory.

    uniform_windows=True instead makes every (view, start) window equally likely.
    Arrays that are pieces of one trajectory (split at a break no window may cross)
    are then sampled as the joined trajectory would be, minus the straddling
    windows; per view, a short piece would otherwise be drawn far more often per
    window than a long one.

    All arrays must share D (feature dim). T is per-array (trajectories of
    different lengths are allowed and common).
    """

    def __init__(self, trajectory_arrays: List[np.ndarray], seq_length=50,
                 samples_per_epoch=128, uniform_windows=False):
        if not trajectory_arrays:
            raise ValueError("MultiTrajectoryDataset needs at least one trajectory array.")
        self.arrays = list(trajectory_arrays)

        # Normalize shape: each array must be 3D [N_i, T_i, D] or 2D [T_i, D] (promoted).
        normalized = []
        for i, arr in enumerate(self.arrays):
            if arr.ndim == 2:
                arr = arr[None, :, :]
            elif arr.ndim != 3:
                raise ValueError(f"Trajectory array {i} must be 2D or 3D; got ndim={arr.ndim}")
            normalized.append(arr)
        self.arrays = normalized

        # Enforce shared D (feature dim). T_i may vary across arrays.
        ref_D = self.arrays[0].shape[2]
        for i, arr in enumerate(self.arrays):
            if arr.shape[2] != ref_D:
                raise ValueError(
                    f"Trajectory array {i} D={arr.shape[2]} does not match "
                    f"first array D={ref_D}. All arrays must share the feature dim."
                )
        self.D = ref_D
        self.T_per_array = np.array([a.shape[1] for a in self.arrays], dtype=np.int64)
        self.T_min = int(self.T_per_array.min())
        self.T_max = int(self.T_per_array.max())
        self.n_per_array = np.array([a.shape[0] for a in self.arrays], dtype=np.int64)
        self.cum_n = np.concatenate([[0], np.cumsum(self.n_per_array)]).astype(np.int64)
        self.N_total = int(self.cum_n[-1])
        self.uniform_windows = bool(uniform_windows)

        self._shared_seq_length = multiprocessing.Value('i', int(seq_length))
        self.samples_per_epoch = int(samples_per_epoch)
        self._validate(seq_length)

    # --- seq_length shared-memory accessors ---

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
        # Every trajectory must be at least seq_len long -- otherwise some
        # selected views would have no valid window.
        if self.T_min < seq_len:
            raise ValueError(
                f"Shortest trajectory T={self.T_min} is less than seq_length {seq_len}"
            )

    # --- Dataset protocol ---

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        current_len = self.seq_length

        if self.uniform_windows:
            n_start = self.T_per_array - current_len + 1
            cum = np.concatenate([[0], np.cumsum(self.n_per_array * n_start)])
            w = int(torch.randint(0, int(cum[-1]), (1,)).item())
            arr_idx = int(np.searchsorted(cum[1:], w, side='right'))
            local_idx, start = divmod(w - int(cum[arr_idx]), int(n_start[arr_idx]))
        else:
            global_idx = int(torch.randint(0, self.N_total, (1,)).item())
            # Locate which underlying array owns global_idx.
            arr_idx = int(np.searchsorted(self.cum_n[1:], global_idx, side='right'))
            local_idx = global_idx - int(self.cum_n[arr_idx])

            T_this = int(self.T_per_array[arr_idx])
            max_start = T_this - current_len
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        window = self.arrays[arr_idx][local_idx, start:start + current_len]
        # mmap-backed slices are read-only and may be non-contiguous; copy into
        # a writable numpy array before handing to torch.from_numpy, then cast.
        return torch.from_numpy(np.ascontiguousarray(window).copy()).float()


# =========================================================================
#  Loaders
# =========================================================================

def _load_augmented_array(path: str) -> np.ndarray:
    """Load an augmented .npy as a memory-mapped array. Accepts the sidecar
    meta.json for cross-checks but does not require it at load time."""
    path = resolve(path)
    arr = np.load(path, mmap_mode='r')
    if arr.ndim != 3:
        raise ValueError(
            f"Augmented file {path} must be 3D [N, T, D]; got shape {arr.shape}."
        )

    meta_path = os.path.splitext(path)[0] + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get('N') is not None and meta['N'] != arr.shape[0]:
            raise ValueError(
                f"{path}: meta.json N={meta['N']} does not match array N={arr.shape[0]}"
            )
    return arr


def first_source_frames(arr: np.ndarray, n: int, path: str) -> List[np.ndarray]:
    """`arr` restricted to the first n frames of the trajectory it was built from.

    A time-reversed view stores the trajectory backwards, so its first n stored frames
    are the LAST n source frames: cutting every view at [:n], as augmented_frame_limits
    does, hands a data-size ablation two disjoint stretches and twice the data it names.
    Forward views are cut at the front, reversed ones at the back. The plan is read from
    the bank's meta.json; views must alternate forward / reversed (balanced_plan does),
    so both halves stay strided mmap views and nothing is read into memory.
    """
    path = resolve(path)
    meta_path = os.path.splitext(path)[0] + '.meta.json'
    if not os.path.exists(meta_path):
        raise ValueError(f"{path}: augmented_source_frames needs {meta_path} to tell reversed views apart")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    rev = np.array([int(r.get('reverse', 0)) for r in meta['augmentation_records']])
    N, T = arr.shape[0], arr.shape[1]
    if len(rev) != N or meta.get('T', T) != T:
        raise ValueError(f"{path}: meta.json describes {len(rev)} views x {meta.get('T')} frames, the array is {N} x {T}")
    if not 0 < n <= T:
        raise ValueError(f"{path}: augmented_source_frames {n} is outside 1..{T}")
    if not rev.any():
        return [arr[:, :n]]
    if not np.array_equal(rev, np.arange(N) % 2):
        raise ValueError(f"{path}: forward and reversed views do not alternate; cannot cut them without copying the bank")
    return [arr[0::2, :n], arr[1::2, T - n:]]


def create_datasets(
    raw_data_matrix=None,
    augmented_paths=None,
    seq_length=50,
    samples_per_epoch=128,
    batch_size=32,
    stats_path="data/norm_stats.json",
):
    """Build a training DataLoader + stats dict.

    Preferred path: pass `augmented_paths` — a list of .npy files from
    archive/scripts/shared/data_prep/generate_augmented_data.py. Loaded via mmap.

    Legacy path: pass a single `raw_data_matrix` ([T, 1 + n_coords] with
    timestamp in column 0). Still random-window sampled per the new Dataset
    API.
    """
    if augmented_paths:
        arrays = [_load_augmented_array(p) for p in augmented_paths]
        train_dataset = MultiTrajectoryDataset(
            arrays,
            seq_length=seq_length,
            samples_per_epoch=samples_per_epoch,
        )
    else:
        if raw_data_matrix is None:
            raise ValueError("Either augmented_paths or raw_data_matrix must be provided.")
        if not torch.is_tensor(raw_data_matrix):
            raw_data = torch.as_tensor(raw_data_matrix, dtype=torch.float32)
        else:
            raw_data = raw_data_matrix.float()
        coords_data = raw_data[:, 1:]  # strip timestamp column
        train_dataset = MolecularSlidingWindowDataset(
            coords_data,
            seq_length=seq_length,
            samples_per_epoch=samples_per_epoch,
        )

    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {'mean': 0.0, 'std': 1.0}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        persistent_workers=True,
        drop_last=True,
        collate_fn=safe_collate,
        pin_memory=False,
    )
    return train_loader, stats
