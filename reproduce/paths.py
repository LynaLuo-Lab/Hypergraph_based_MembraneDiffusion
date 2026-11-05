"""Where the reproduction scripts find the MD data and the downloaded assets and where they write."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next(d for d in (os.path.abspath(HERE + '/..' * n) for n in range(8)) if os.path.isfile(d + '/pyproject.toml'))
MD_DIR = os.path.join(ROOT, 'data/membrane/half')
MESH_FRAME = os.path.join(ROOT, 'data/membrane/mesh/7piezo_traj1_frame4262.npz')
ASSETS = os.path.join(ROOT, 'assets')
CKPT = os.path.join(ASSETS, 'checkpoints')
ROLLS = os.path.join(ASSETS, 'rollouts/heldout')
ROLLS40K = os.path.join(ASSETS, 'rollouts/forecast')
HYPER = os.path.join(ASSETS, 'videos')
RESULTS = os.path.join(ROOT, 'results')
TRACES = os.path.join(RESULTS, 'traces')
FIGURES = os.path.join(ROOT, 'figures')
VIDEOS = os.path.join(ROOT, 'videos')
CACHE = os.path.join(ROOT, 'cache')
LOGS = os.path.join(ROOT, 'logs')
SELECTION = os.path.join(HERE, 'selection_nd.json')

RUNS = ('7piezo_R1', '7piezo_R2', 'piezo14_rep01', 'piezo14_rep02')
SECOND = [f'{r}second.csv' for r in RUNS] + [f'membrane_train_{r}_arms.npz' for r in
          ('7piezo_R1second_part0', '7piezo_R1second_part1', '7piezo_R2second', 'piezo14_rep01second', 'piezo14_rep02second')]
FIRST = [f'{r}half.csv' for r in RUNS] + [f'membrane_train_{r}half_arms.npz' for r in RUNS]


def check_md(first=False):
    """raise listing the missing MD files; the first halves are needed to build a model from a checkpoint, to select epochs and to train"""
    miss = [f for f in SECOND + (FIRST if first else []) if not os.path.exists(os.path.join(MD_DIR, f))]
    if miss:
        raise SystemExit(f'missing MD files in {MD_DIR}:\n  ' + '\n  '.join(miss))


def check_assets(*parts):
    miss = [p for p in (parts or ('checkpoints', 'rollouts/heldout', 'rollouts/forecast')) if not os.path.isdir(os.path.join(ASSETS, p))]
    if miss:
        raise SystemExit(f'missing assets {miss} in {ASSETS}; run scripts/fetch_assets.py')
