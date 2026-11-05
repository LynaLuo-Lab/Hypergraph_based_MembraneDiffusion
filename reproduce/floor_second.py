"""MD-vs-MD g(r) floor on the second halves (gr_lib.Ref.score of one MD trajectory against the other). Writes results/floor_second.json."""
import json
import os

import gr_lib as G
from paths import RESULTS

SECOND = {7: ['data/membrane/half/7piezo_R1second.csv', 'data/membrane/half/7piezo_R2second.csv'],
          14: ['data/membrane/half/piezo14_rep01second.csv', 'data/membrane/half/piezo14_rep02second.csv']}
out = {'code': 'gr_lib.Ref.score (KL(MD||other MD)/KL(MD||random), merge 1 for 7P, 20 for 14P)'}
for P in (7, 14):
    G.GT[P] = SECOND[P]
    R = G.Ref(P)
    a, b = R.names
    out.setdefault('second', {})[f'{P}P'] = {a: R.score(a, R.H[b]), b: R.score(b, R.H[a]), 'merge': R.f, 'inputs': SECOND[P]}
    print('second', P, a, f'{R.score(a, R.H[b]):.3f}', b, f'{R.score(b, R.H[a]):.3f}')
json.dump(out, open(os.path.join(RESULTS, 'floor_second.json'), 'w'), indent=1)
