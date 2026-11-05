"""Shared setup for the held-out write-up: the two systems, the checkpoint picks of selection_nd.json (one per run, chosen on the first
halves), and the second-half MD, node depth and start angles the held-out rollouts are scored against. 7-PIEZO keeps runs with n_epochs >= 20."""
import json
import os
import sys

from paths import CKPT, HERE, ROOT, SELECTION, check_md

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
from membrane.FitObjectiveCallback import depth_std, load_depth, load_md, start_thetas

N, SEED, MIN_EPOCHS = 32, 3000, 20
SQ_OFF = {'squash_enabled': False, 'w_squash': 0.0}
D = 'data/membrane/half/'
SYSTEMS = {
    '7-PIEZO': dict(prefix='h7', cfg='configs/7piezo/full.json', nm=0.4018202, n_forecast=8,
                    gt=[D + '7piezo_R1second.csv', D + '7piezo_R2second.csv'],
                    npz=[D + 'membrane_train_7piezo_R1second_part0_arms.npz', D + 'membrane_train_7piezo_R1second_part1_arms.npz',
                         D + 'membrane_train_7piezo_R2second_arms.npz']),
    '14-PIEZO': dict(prefix='h14', cfg='configs/14piezo/full.json', nm=0.432600314049058, n_forecast=5,
                     gt=[D + 'piezo14_rep01second.csv', D + 'piezo14_rep02second.csv'],
                     npz=[D + 'membrane_train_piezo14_rep01second_arms.npz', D + 'membrane_train_piezo14_rep02second_arms.npz']),
}
NAMES = {'7piezo_R1second': 'trajectory 1', '7piezo_R2second': 'trajectory 2',
         'piezo14_rep01second': 'replica 1', 'piezo14_rep02second': 'replica 2'}


def heldout_lk():
    import heldout_lk as H
    return H


def picks(system, kinds=('sq', 'nosq', 'base', 'lk')):
    """checkpoints to score: dicts with run, seed, epoch, n_epochs, variant ('sq' final with squash, 'nosq' final with the
    head off, 'base' baseline), tag, path, overrides. Read from selection_nd.json at call time, so a refreshed selection
    gives new epochs (and new tags)."""
    p = SYSTEMS[system]['prefix']
    sel = json.load(open(SELECTION))
    out = []
    for run in sorted(sel):
        arm, seed = run.rsplit('_s', 1)
        if not run.startswith(p + '_') or (p == 'h7' and sel[run]['n_epochs'] < MIN_EPOCHS):
            continue
        e = sel[run]['epoch']
        path = os.path.join(CKPT, run, f'epoch-epoch={e:03d}.ckpt')
        lk = arm.endswith('lk_both')
        if not os.path.isfile(os.path.join(CKPT, run, 'config.json')):
            raise SystemExit(f'missing {os.path.join(CKPT, run)}; run scripts/fetch_assets.py')
        for v, over in (('sq', {'squash_enabled': True}), ('nosq', SQ_OFF)) if arm.endswith('lk_both_sq') else \
                (('lk', None),) if lk else (('base', None),):
            if v in kinds:
                out.append(dict(system=system, run=run, seed=int(seed), epoch=e, n_epochs=sel[run]['n_epochs'], variant=v,
                                tag=f'{run}_{v}_e{e:03d}', path=path, overrides=over))
    return out


def forecast_picks(system):
    """final runs with squash on for the forecast: the n_forecast with most epochs trained (ties to the lower seed)"""
    f = picks(system, ('sq',))
    return sorted(f, key=lambda r: (-r['n_epochs'], r['seed']))[:SYSTEMS[system]['n_forecast']]


def reference(system):
    """second-half MD, node depth, start angles and box of a system"""
    check_md()
    s = SYSTEMS[system]
    box = float(json.load(open(os.path.join(ROOT, s['cfg'])))['box_size'])
    md = load_md(s['gt'])
    depth, thetas = load_depth(md, s['npz'], box), start_thetas(md, s['npz'], box)
    assert set(depth) == set(md) == set(thetas), (set(depth), set(md), set(thetas))
    return md, depth, thetas, box

