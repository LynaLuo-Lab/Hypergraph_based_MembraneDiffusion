import json
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.paths import resolve, resolve_all
from data.DataSet import (
    MolecularSlidingWindowDataset,
    MultiTrajectoryDataset,
    _load_augmented_array,
    first_source_frames,
    safe_collate,
)


class BrownianDataModule(L.LightningDataModule):
    """Dispatches between three data-source modes:

    1. augmented_dataset_paths: list of pre-augmented .npy files (preferred).
       Loaded via mmap; random (view, start) sampling per __getitem__.

    2. dataset_paths: list of raw CSV trajectories. Loaded into torch tensors
       (still in-memory), wrapped in MultiTrajectoryDataset with one view each.

    3. dataset_path: single raw CSV trajectory (legacy single-trajectory mode).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.augmented_paths = resolve_all(config.get('augmented_dataset_paths', []) or [])
        self.data_path = resolve(config.get('dataset_path', ''))
        self.data_paths = resolve_all(config.get('dataset_paths', []) or [])
        self.samples_per_epoch = int(config.get('samples_per_epoch', 128))
        self.num_workers = int(config.get('num_workers', 16))
        self.multi_system = config.get('multi_system', False)
        self.membrane_hg = config.get('membrane_hypergraph', False)

        self.current_seq_length = config.get('seq_len_start', 50)
        self.train_dataset = None
        self.stats = None

    def setup(self, stage=None):
        if self.membrane_hg:
            # membrane_data_paths (plural, a list) pools several trajectories so a
            # batch mixes ICs from all of them; membrane_data_path (singular) is
            # the original single-trajectory path.
            npz_paths = self.config.get('membrane_data_paths') or None
            if npz_paths:
                from data.membrane_dataset import MultiMembraneWindowDataset
                self.train_dataset = MultiMembraneWindowDataset(
                    npz_paths=npz_paths,
                    seq_length=self.current_seq_length,
                    samples_per_epoch=self.samples_per_epoch,
                    augment=self.config.get('membrane_augment', True),
                    n_particles=self.config.get('n_particles', 7),
                    seed=self.config.get('seed', 0),
                    balance=self.config.get('membrane_pool_balance', False),
                    return_theta=self.config.get('arms_enabled', False),
                    return_blade=self.config.get('geom_blade_enabled', False),
                    augment_translate=self.config.get('membrane_augment_translate', True),
                    augment_permute=self.config.get('membrane_augment_permute', True),
                    species_groups=self.config.get('species_groups'),
                )
                self.stats = {'mean': 0.0, 'std': 1.0}
                from data.membrane_dataset import check_membrane_system
                check_membrane_system(self.train_dataset, self.config)
                print(f"  Membrane-hypergraph POOL: {self.train_dataset.n_traj} trajectories "
                      f"{self.train_dataset.Ns.tolist()} = {self.train_dataset.N} frames, "
                      f"balance={self.train_dataset.balance}, augment={self.train_dataset.augment}, "
                      f"samples_per_epoch={self.samples_per_epoch}, seq_len={self.current_seq_length}")
                return
            from data.membrane_dataset import MembraneWindowDataset
            self.train_dataset = MembraneWindowDataset(
                npz_path=self.config['membrane_data_path'],
                seq_length=self.current_seq_length,
                samples_per_epoch=self.samples_per_epoch,
                augment=self.config.get('membrane_augment', True),
                n_particles=self.config.get('n_particles', 7),
                seed=self.config.get('seed', 0),
                return_theta=self.config.get('arms_enabled', False),
                return_blade=self.config.get('geom_blade_enabled', False),
                augment_translate=self.config.get('membrane_augment_translate', True),
                augment_permute=self.config.get('membrane_augment_permute', True),
                species_groups=self.config.get('species_groups'),
            )
            self.stats = {'mean': 0.0, 'std': 1.0}
            from data.membrane_dataset import check_membrane_system
            check_membrane_system(self.train_dataset, self.config)
            print(f"  Membrane-hypergraph mode: N={self.train_dataset.N} frames, "
                  f"augment={self.train_dataset.augment}, "
                  f"samples_per_epoch={self.samples_per_epoch}, seq_len={self.current_seq_length}")
            return

        if self.multi_system:
            from data.multi_system import MultiSystemDataset
            system_configs = self.config['system_configs']
            systems = []
            for sc in system_configs:
                arrays = [_load_augmented_array(p) for p in sc['augmented_dataset_paths']]
                for arr in arrays:
                    print(f"  Multi-system [{sc['name']}]: {arr.shape} (mmap)")
                # Concatenate multiple augmented files along the views axis (axis 0)
                combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
                systems.append({
                    'name': sc['name'],
                    'n_particles': sc['n_particles'],
                    'box_size': sc['box_size'],
                    'array': combined,
                })
            self.train_dataset = MultiSystemDataset(
                systems,
                seq_length=self.current_seq_length,
                samples_per_epoch=self.samples_per_epoch,
            )
            self.stats = {'mean': 0.0, 'std': 1.0}
            print(f"  Multi-system mode: {len(systems)} systems, "
                  f"samples_per_epoch={self.samples_per_epoch}")
            return

        if self.augmented_paths:
            arrays = []
            # Optional per-file cap on GT source frames (the T axis), for data-size
            # ablations. augmentation (the views axis) is untouched. mmap slice = free.
            frame_limits = self.config.get('augmented_frame_limits', None) or []
            # The first N frames of the SOURCE trajectory, whichever way a view runs
            # (first_source_frames). The cap above slices the stored axis and is wrong
            # for a bank with time-reversed views.
            source_frames = self.config.get('augmented_source_frames', None) or []
            for idx, path in enumerate(self.augmented_paths):
                arr = _load_augmented_array(path)
                lim = frame_limits[idx] if idx < len(frame_limits) else None
                src = source_frames[idx] if idx < len(source_frames) else None
                if src:
                    if lim:
                        raise ValueError(f"{path}: set augmented_frame_limits or augmented_source_frames, not both")
                    parts = first_source_frames(arr, int(src), path)
                    arrays.extend(parts)
                    print(f"  Augmented trajectory: {path} -> first {int(src)} source frames, "
                          + ' + '.join(str(q.shape) for q in parts) + " (mmap)")
                    continue
                if lim:
                    arr = arr[:, :int(lim), :]
                arrays.append(arr)
                print(f"  Augmented trajectory: {path} -> shape {arr.shape} (mmap)"
                      + (f" [frame-cap {int(lim)}]" if lim else ""))
            self.train_dataset = MultiTrajectoryDataset(
                arrays,
                seq_length=self.current_seq_length,
                samples_per_epoch=self.samples_per_epoch,
                uniform_windows=self.config.get('uniform_windows', False),
            )
            print(
                f"  Augmented-mode: N_total={self.train_dataset.N_total} views, "
                f"T range=[{self.train_dataset.T_min}, {self.train_dataset.T_max}], "
                f"samples_per_epoch={self.samples_per_epoch}, "
                f"seq_len={self.current_seq_length}"
            )
        elif self.data_paths:
            tensors = []
            for path in self.data_paths:
                raw = (
                    pd.read_csv(path, sep=None, engine='python', header=None)
                    .apply(pd.to_numeric, errors='coerce')
                    .dropna()
                    .values
                )
                t = torch.tensor(raw).float()[:, 1:]  # skip timestamp
                tensors.append(t.numpy()[None, :, :])  # promote to [1, T, D]
                print(f"  Loaded trajectory: {path} -> {t.shape}")
            self.train_dataset = MultiTrajectoryDataset(
                tensors,
                seq_length=self.current_seq_length,
                samples_per_epoch=self.samples_per_epoch,
                uniform_windows=self.config.get('uniform_windows', False),
            )
            print(
                f"  Raw multi-trajectory mode: {len(tensors)} trajectories, "
                f"samples_per_epoch={self.samples_per_epoch}, seq_len={self.current_seq_length}"
            )
        else:
            raw_data = (
                pd.read_csv(self.data_path, sep=None, engine='python', header=None)
                .apply(pd.to_numeric, errors='coerce')
                .dropna()
                .values
            )
            raw_tensor = torch.tensor(raw_data).float()
            coords_data = raw_tensor[:, 1:]
            self.train_dataset = MolecularSlidingWindowDataset(
                coords_data,
                seq_length=self.current_seq_length,
                samples_per_epoch=self.samples_per_epoch,
            )
            print(
                f"  Single-trajectory mode: T={len(coords_data)}, "
                f"samples_per_epoch={self.samples_per_epoch}, seq_len={self.current_seq_length}"
            )

        # The data's particle count must match the model's. Nothing checked this: a
        # config whose augmented_dataset_paths point at another system loads happily
        # and only fails deep in the rollout as
        # "size of tensor a (21) must match tensor b (6)" -- 7 particles against 2.
        # With several systems in one study a mismatched path is a live hazard, and a
        # wrong-data run that merely LOOKED plausible would be worse than a crash.
        self._check_particle_width()

        stats_path = resolve(self.config.get('stats_path', 'data/norm_stats.json'))
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {'mean': 0.0, 'std': 1.0}

    def _check_particle_width(self):
        """Assert the loaded feature width is n_particles * 3."""
        if self.multi_system or self.membrane_hg:
            return                      # these carry per-system widths of their own
        ds = self.train_dataset
        if ds is None:
            return
        D = None
        arrays = getattr(ds, 'arrays', None)
        if arrays:
            D = arrays[0].shape[-1]
        elif getattr(ds, 'data', None) is not None:
            D = ds.data.shape[-1]
        if D is None:
            return
        n = self.config.get('n_particles', 7)
        if D != n * 3:
            raise ValueError(
                f"Loaded trajectory width D={D} but n_particles={n} implies D={n * 3}. "
                f"The config's data paths and its particle count disagree -- check "
                f"augmented_dataset_paths / dataset_paths for "
                f"{self.config.get('dataset_path')!r}. Training would fail deep in the "
                f"rollout with an unhelpful shape error, or silently learn the wrong "
                f"system.")

    def set_train_seq_length(self, length):
        """Called by the curriculum callback to update state."""
        self.current_seq_length = int(length)
        if self.train_dataset is not None:
            self.train_dataset.update_n_samples(self.current_seq_length)

    def train_dataloader(self):
        collate_fn = safe_collate
        if self.multi_system:
            from data.multi_system import multi_system_collate
            collate_fn = multi_system_collate
        elif self.membrane_hg:
            from data.membrane_dataset import membrane_collate
            collate_fn = membrane_collate
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )
