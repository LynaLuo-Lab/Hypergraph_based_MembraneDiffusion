"""histogram score of the saved held-out rollouts (rolls/) against each second-half MD trajectory, gr_lib.Ref.score as
heldout_second.py (KL(MD||set) of the pair-distance histograms over KL(MD||random); 0 = MD, 1 = random), next to its floor
from results/floor_second.json (the other second half scored the same way). One row per tag in
results/g_half.jsonl; tags already there are skipped.
usage: score_histogram.py"""
import json
import os

import numpy as np

import gr_lib as G
from common import SEED, SYSTEMS, picks, reference
from paths import RESULTS, ROLLS

OUTF = os.path.join(RESULTS, 'g_half.jsonl')
if __name__ == '__main__':
    done = {json.loads(l)['tag'] for l in open(OUTF)} if os.path.exists(OUTF) else set()
    floors = json.load(open(os.path.join(RESULTS, 'floor_second.json')))['second']
    for system, s in SYSTEMS.items():
        P = int(s['prefix'][1:])
        md, _, _, box = reference(system)
        G.GT[P] = s['gt']
        ref = G.Ref(P)
        assert ref.names == list(md) and abs(ref.box - box) < 1e-6
        fl = floors[f'{P}P']
        for p in picks(system):
            f = os.path.join(ROLLS, f"{p['tag']}_seed{SEED}.npz")
            if p['tag'] in done or not os.path.exists(f):
                continue
            z = np.load(f)
            g = {k: ref.score(k, G.hist(z[f'xy_{k}'], box)) for k in md}
            with open(OUTF, 'a') as fh:
                fh.write(json.dumps({'tag': p['tag'], 'g': g, 'g_floor': {k: fl[k] for k in md}}) + '\n')
            print(p['tag'], {k[-8:]: round(v, 3) for k, v in g.items()}, flush=True)
