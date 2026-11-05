"""Parameter counts by module of the full model, 7-PIEZO and 14-PIEZO, from the first selected checkpoint of each system's full-model runs.
usage: param_count.py"""

import torch

from common import SYSTEMS, picks
from paths import check_md
from scripts.membrane.eval.sample_membrane import load_membrane_checkpoint

check_md(first=True)
for name in SYSTEMS:
    p = picks(name, ('sq',))[0]
    model, _ = load_membrane_checkpoint(p['path'], 'cpu', p['overrides'])
    print(f"== {name} ({p['tag']})")
    tot = 0
    for n, m in model.named_children():
        c = sum(x.numel() for x in m.parameters())
        tot += c
        print(f'{n:28s} {c:>10,d}')
    print(f"{'total':28s} {sum(x.numel() for x in model.parameters()):>10,d}  (children sum {tot:,d})")
