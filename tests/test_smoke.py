import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'reproduce'))
import paths

CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'configs/*/*.json')))


def test_imports():
    import membrane.model_multilag
    import membrane.model_registry
    import nbcore.config.validate
    import nbcore.sophia_g
    import scripts.membrane.eval.sample_membrane


def test_configs_validate():
    from nbcore.config.validate import validate_config
    assert len(CONFIGS) == 6
    for f in CONFIGS:
        c = json.load(open(f))
        validate_config(c, f)
        assert c['default_root_dir'].startswith('runs/')


def test_paths_contract():
    assert paths.ROOT == ROOT
    assert paths.MD_DIR == os.path.join(ROOT, 'data/membrane/half')
    assert paths.MESH_FRAME.endswith('data/membrane/mesh/7piezo_traj1_frame4262.npz')
    assert paths.CKPT == os.path.join(ROOT, 'assets/checkpoints')
    assert paths.ROLLS == os.path.join(ROOT, 'assets/rollouts/heldout')
    assert paths.ROLLS40K == os.path.join(ROOT, 'assets/rollouts/forecast')
    assert paths.HYPER == os.path.join(ROOT, 'assets/videos')
    assert (paths.RESULTS, paths.FIGURES, paths.VIDEOS) == tuple(os.path.join(ROOT, d) for d in ('results', 'figures', 'videos'))


def test_config_data_paths_are_shipped():
    for f in CONFIGS:
        c = json.load(open(f))
        for p in c['stability_gt_paths'] + c['membrane_data_paths']:
            assert os.path.basename(p) in paths.FIRST, p


def test_check_md_names_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, 'MD_DIR', str(tmp_path))
    with pytest.raises(SystemExit, match='7piezo_R1second.csv'):
        paths.check_md()


def test_selection_runs_have_configs():
    sel = json.load(open(paths.SELECTION))
    assert len(sel) == 38
    for run in sel:
        assert run.startswith(('h7_', 'h14_'))


def _first_checkpoint():
    cks = sorted(glob.glob(os.path.join(paths.CKPT, '*/epoch-epoch=*.ckpt')))
    if not cks:
        pytest.skip('no checkpoints in assets/checkpoints (run scripts/fetch_assets.py)')
    if not os.path.isdir(paths.MD_DIR) or not all(os.path.exists(os.path.join(paths.MD_DIR, f)) for f in paths.FIRST):
        pytest.skip('first-half MD files missing in data/membrane/half')
    return cks[0]


def test_checkpoint_rollout_step():
    import torch
    from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint
    ck = _first_checkpoint()
    os.chdir(ROOT)
    model, cfg = load_membrane_checkpoint(ck, 'cpu')
    model.eval()
    P, box = int(cfg['n_particles']), float(model.box_size)
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(1, 1, P * 3, generator=g) * box
    with torch.no_grad():
        out = model.predict_future(x0, steps=1, return_unwrapped=False, initial_theta=torch.rand(1, P, generator=g) * 6.28)
    x = (out[0] if isinstance(out, tuple) else out)
    assert x.shape[-1] == P * 3 and torch.isfinite(x).all()
