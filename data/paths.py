"""Where a data file lives now, given any path it has had.

Data paths are frozen as plain strings in checkpoint hyper_parameters, sweeps/*.json and k8s jobs pinned at old
commits. MOVED maps each old repo-relative path to its current one so those keep resolving. Append only; a key ending
in '/' moves a folder. resolve() returns a path that exists unchanged and looks only a missing one up, so nothing that
works today behaves differently. Stdlib only: k8s/link_shared_data.py imports this before anything is installed.

    python data/paths.py migrate [--apply]     move this machine's old-layout files to where MOVED says they live
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MOVED = {
    'data/7piezo_R1/': 'data/shared/7piezo/marker_dumps/7piezo_R1/',
    'data/7piezo_centers_2_trimmed.csv': 'data/shared/7piezo/raw/7piezo_centers_2_trimmed.csv',
    'data/7piezo_extension/': 'data/shared/7piezo/marker_dumps/7piezo_extension/',
    'data/7piezo_traj1_aug40320.meta.json': 'data/shared/7piezo/augmented/7piezo_traj1_aug40320.meta.json',
    'data/7piezo_traj1_aug40320.npy': 'data/shared/7piezo/augmented/7piezo_traj1_aug40320.npy',
    'data/7piezo_traj1_aug8_d4.meta.json': 'data/shared/7piezo/augmented/7piezo_traj1_aug8_d4.meta.json',
    'data/7piezo_traj1_aug8_d4.npy': 'data/shared/7piezo/augmented/7piezo_traj1_aug8_d4.npy',
    'data/7piezo_traj2_aug24k.meta.json': 'data/shared/7piezo/augmented/7piezo_traj2_aug24k.meta.json',
    'data/7piezo_traj2_aug24k.npy': 'data/shared/7piezo/augmented/7piezo_traj2_aug24k.npy',
    'data/7piezo_traj2_aug40320.meta.json': 'data/shared/7piezo/augmented/7piezo_traj2_aug40320.meta.json',
    'data/7piezo_traj2_aug40320.npy': 'data/shared/7piezo/augmented/7piezo_traj2_aug40320.npy',
    'data/7piezo_traj2_aug8_d4.meta.json': 'data/shared/7piezo/augmented/7piezo_traj2_aug8_d4.meta.json',
    'data/7piezo_traj2_aug8_d4.npy': 'data/shared/7piezo/augmented/7piezo_traj2_aug8_d4.npy',
    'data/7piezo_traj2_dedup_aug24k.meta.json': 'data/shared/7piezo/augmented/7piezo_traj2_dedup_aug24k.meta.json',
    'data/7piezo_traj2_dedup_aug24k.npy': 'data/shared/7piezo/augmented/7piezo_traj2_dedup_aug24k.npy',
    'data/7piezo_traj2_ext/': 'data/shared/7piezo/marker_dumps/7piezo_traj2_ext/',
    'data/7piezo_traj2_first3k.csv': 'data/shared/7piezo/trajectories/7piezo_traj2_first3k.csv',
    'data/7piezo_traj2_first3k_aug40320.npy': 'data/shared/7piezo/augmented/7piezo_traj2_first3k_aug40320.npy',
    'data/7piezo_trek_v2/': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/',
    'data/7piezo_trek_v2/trek_centers_norm_11k_dedup_timesteps.txt': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/timesteps/trek_centers_norm_11k_dedup_timesteps.txt',
    'data/7piezo_trek_v2/trek_centers_norm_11k_dedup_unwrapped_angstrom.csv': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/angstrom/trek_centers_norm_11k_dedup_unwrapped_angstrom.csv',
    'data/7piezo_trek_v2/trek_centers_norm_11k_timesteps.txt': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/timesteps/trek_centers_norm_11k_timesteps.txt',
    'data/7piezo_trek_v2/trek_centers_norm_11k_unwrapped_angstrom.csv': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/angstrom/trek_centers_norm_11k_unwrapped_angstrom.csv',
    'data/7piezo_trek_v2/trek_centers_norm_11k_wrapped_angstrom.csv': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/angstrom/trek_centers_norm_11k_wrapped_angstrom.csv',
    'data/7piezo_trek_v2/trek_centers_rep2_timesteps.txt': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/timesteps/trek_centers_rep2_timesteps.txt',
    'data/7piezo_trek_v2/trek_centers_rep2_unwrapped_angstrom.csv': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/angstrom/trek_centers_rep2_unwrapped_angstrom.csv',
    'data/7piezo_trek_v2/trek_centers_rep3_timesteps.txt': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/timesteps/trek_centers_rep3_timesteps.txt',
    'data/7piezo_trek_v2/trek_centers_rep3_unwrapped_angstrom.csv': 'data/diffusion/piezo_trek/alternate_reextraction_2026-10-22/v2/angstrom/trek_centers_rep3_unwrapped_angstrom.csv',
    'data/GM_1010_v2/': 'archive/data/diffusion/gm1010/v2/',
    'data/GM_1010_v3/': 'archive/data/diffusion/gm1010/v3/',
    'data/GM_1010_v5/': 'data/diffusion/gm1010/alternate_reextraction_2026-10-22/v5/',
    'data/combine_and_normalize.py': 'data/shared/7piezo/build/combine_and_normalize.py',
    'data/data_centers_xyz.csv': 'archive/data/shared/7piezo/legacy/data_centers_xyz.csv',
    'data/data_centers_xyz_unwrapped_angstrom.csv': 'archive/data/shared/7piezo/legacy/data_centers_xyz_unwrapped_angstrom.csv',
    'data/data_full.csv': 'archive/data/shared/7piezo/legacy/data_full.csv',
    'data/get_wandb_logs.py': 'archive/data/get_wandb_logs.py',
    'data/membrane/footprint_A.npz': 'data/membrane/footprint/footprint_A.npz',
    'data/membrane/footprint_A.png': 'data/membrane/footprint/footprint_A.png',
    'data/membrane/footprint_explore.npz': 'data/membrane/footprint/footprint_explore.npz',
    'data/membrane/footprint_explore.png': 'data/membrane/footprint/footprint_explore.png',
    'data/membrane/membrane_train.npz': 'archive/data/membrane/membrane_train.npz',
    'data/membrane/membrane_train_2traj.npz': 'archive/data/membrane/membrane_train_2traj.npz',
    'data/membrane/membrane_train_7piezo_traj2_part0_arms.npz.PRESPLIT.bak': 'archive/data/membrane/membrane_train_7piezo_traj2_part0_arms.npz.PRESPLIT.bak',
    'data/membrane/membrane_train_7piezo_traj2_part1_arms.npz.PRESPLIT.bak': 'archive/data/membrane/membrane_train_7piezo_traj2_part1_arms.npz.PRESPLIT.bak',
    'data/membrane/membrane_train_arms.npz': 'archive/data/membrane/membrane_train_arms.npz',
    'data/membrane/membrane_train_traj1.npz': 'archive/data/membrane/membrane_train_traj1.npz',
    'data/normalize_data.py': 'data/shared/7piezo/build/normalize_data.py',
    'data/piezo1/': 'data/shared/piezo1/',
    'data/piezo14/': 'data/diffusion/piezo14/processed/',
    'data/piezo14_rep01/': 'data/diffusion/piezo14/marker_dumps/rep01/',
    'data/piezo14_rep02/': 'data/diffusion/piezo14/marker_dumps/rep02/',
    'data/piezo2/': 'data/diffusion/piezo2/',
    'data/track1_1_CDDUW.dat': 'data/shared/7piezo/raw/track1_1_CDDUW.dat',
    'data/track2_1_CDDUW.dat': 'data/shared/7piezo/raw/track2_1_CDDUW.dat',
    'data/track3_1_CDDUW.dat': 'data/shared/7piezo/raw/track3_1_CDDUW.dat',
    'data/track4_1_CDDUW.dat': 'data/shared/7piezo/raw/track4_1_CDDUW.dat',
    'data/track5_1_CDDUW.dat': 'data/shared/7piezo/raw/track5_1_CDDUW.dat',
    'data/track6_1_CDDUW.dat': 'data/shared/7piezo/raw/track6_1_CDDUW.dat',
    'data/track7_1_CDDUW.dat': 'data/shared/7piezo/raw/track7_1_CDDUW.dat',
    'membrane_loss_calibration.json': 'data/shared/calibration/membrane_loss_calibration.json',
    'stab_calibration.json': 'data/shared/calibration/stab_calibration.json',
}

PATH_KEYS = ('dataset_path', 'dataset_paths', 'augmented_dataset_paths', 'membrane_data_path', 'membrane_data_paths',
             'membrane_mesh_path', 'stability_gt_paths', 'stability_floor_json', 'stats_path')


def moved(rel):
    """repo-relative path -> where MOVED sends it, longest key first, chains followed"""
    rel = os.path.normpath(rel)
    order = sorted(MOVED, key=len, reverse=True)
    for _ in range(16):
        for old in order:
            o = old.rstrip('/')
            if rel == o or (old.endswith('/') and rel.startswith(old)):
                rel = MOVED[old].rstrip('/') + rel[len(o):]
                break
        else:
            return rel
    raise RuntimeError(f'MOVED has a cycle at {rel}')


def resolve(path):
    if not isinstance(path, str) or not path or os.path.exists(path):
        return path
    ap = os.path.abspath(path)
    rel = os.path.relpath(ap, ROOT)
    if rel.startswith('..'):
        return path
    new = os.path.join(ROOT, moved(rel))
    if new == ap or not os.path.exists(new):
        return path
    return new if os.path.isabs(path) else os.path.relpath(new)


def resolve_all(p):
    """a path, or nested lists/tuples of them (stability_gt_paths), shape and type kept"""
    return type(p)(resolve_all(x) for x in p) if isinstance(p, (list, tuple)) else resolve(p)


def resolve_config(config):
    """the data paths of a config, in place"""
    for k in PATH_KEYS:
        if config.get(k) is not None:
            config[k] = resolve_all(config[k])
    for sc in config.get('system_configs') or ():
        if isinstance(sc, dict) and sc.get('augmented_dataset_paths') is not None:
            sc['augmented_dataset_paths'] = resolve_all(sc['augmented_dataset_paths'])
    return config


def migrate(apply=False):
    """old-layout files still on this machine -> their MOVED homes; never overwrites"""
    plan = []
    for old in sorted(MOVED, key=len):
        src, dst = os.path.join(ROOT, old.rstrip('/')), os.path.join(ROOT, moved(old))
        if os.path.lexists(src) and not os.path.lexists(dst):
            plan.append((src, dst))
        elif os.path.lexists(src):
            print(f'  both exist, left alone: {src} and {dst}')
    for src, dst in plan:
        print(f'  {"move" if apply else "would move"} {os.path.relpath(src, ROOT)} -> {os.path.relpath(dst, ROOT)}')
        if apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
    print(f'{len(plan)} {"moved" if apply else "to move (dry run; --apply moves them)"}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['migrate'])
    ap.add_argument('--apply', action='store_true')
    migrate(ap.parse_args().apply)
