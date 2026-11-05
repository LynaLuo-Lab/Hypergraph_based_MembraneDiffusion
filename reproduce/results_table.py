"""RESULTS_half.md for the write-up writer: every number the Results section may use for the half-trajectory models on the
held-out second halves, per training run and summarised (mean, sd over runs, n), with the other-second-half and random
references, the final-vs-baseline and squash-on-vs-off comparisons (two-sided Mann-Whitney p per metric), the κ
calibration on the second halves and the 40,000-frame forecast. Reads rows_half.jsonl, stats_half.json (figures.py) and
forecast_stats_half.json (forecast.py, optional). Writes results/RESULTS_half.md.
usage: results_table.py"""
import json
import os

import numpy as np
from scipy import stats as sps

from common import NAMES, SEED, SYSTEMS, picks
from paths import RESULTS, ROLLS, ROOT

OUT = os.path.join(RESULTS, 'RESULTS_half.md')
st = json.load(open(os.path.join(RESULTS, 'stats_half.json')))
FC = os.path.join(RESULTS, 'forecast_stats_half.json')
fc = json.load(open(FC)) if os.path.exists(FC) else {}
rows = {}
for r in map(json.loads, open(os.path.join(RESULTS, 'rows_half.jsonl'))):
    rows[r['tag']] = r
BIG_MIN = 0.5          # the largest cluster is read where MD differs from random placement by at least half a protein
gr = {}
GF = os.path.join(RESULTS, 'g_half.jsonl')
for r in map(json.loads, open(GF)):
    gr[r['tag']] = r
L = []
w = L.append


def at(path, needle, nth=1):
    """path:line of the nth line containing needle, so the definitions below cite the code as it is now"""
    n = 0
    for i, line in enumerate(open(os.path.join(ROOT, path)), 1):
        if needle in line:
            n += 1
            if n == nth:
                return f'{path}:{i}'
    raise ValueError((path, needle))


FO, GRL = 'membrane/fit_objective.py', 'reproduce/gr_lib.py'
LK, FH = 'reproduce/heldout_lk.py', 'reproduce/'


def ms(v, f='{:.3f}'):
    v = np.asarray(v, float)
    return (f + ' +/- ' + f.replace('+', '')).format(v.mean(), v.std(ddof=1)) if len(v) > 1 else f.format(v[0])


def get(system, v):
    """rows of one variant in seed order, joined to the current picks on the tag and checked on (run, seed)"""
    out = []
    for p in sorted(picks(system, (v,)), key=lambda p: p['seed']):
        r = rows[p['tag']]
        assert (r['run'], r['seed'], r['epoch']) == (p['run'], p['seed'], p['epoch']), p['tag']
        out.append({**r, 'n_epochs': p['n_epochs']})   # epochs trained from the current selection, the row tag is reused
    return out


def mwu(a, b):
    return sps.mannwhitneyu(a, b, alternative='two-sided').pvalue


def cmp_table(title, A, B, la, lb, metrics, paired=False):
    w(f'\n{title}\n')
    if paired:
        assert [r['seed'] for r in A] == [r['seed'] for r in B]
    w(f'| metric | n {la} | n {lb} | {la} | {lb} | Mann-Whitney p |' + (' Wilcoxon paired p |' if paired else ''))
    w('|---|---|---|---|---|---|' + ('---|' if paired else ''))
    for name, f, fmt in metrics:
        a, b = np.array([f(r) for r in A]), np.array([f(r) for r in B])
        w(f'| {name} | {len(a)} | {len(b)} | {ms(a, fmt)} | {ms(b, fmt)} | {mwu(a, b):.3f} |'
          + (f' {sps.wilcoxon(a, b).pvalue:.3f} |' if paired else ''))


w('# Results, held-out second halves\n')
w('Protocol. The models are trained on the first half (in time) of each MD trajectory and evaluated on the held-out second halves. '
  'Per run, one checkpoint: the epoch that minimises the in-training objective on the first halves, taken as clustering excess + depth '
  'excess + overlap guard (no diffusion term); nothing is selected on the second halves. 7-PIEZO uses only runs with at least 20 epochs '
  'trained; 14-PIEZO uses all runs. Evaluation: 32 free rollouts per second-half trajectory from its frame 0, run for its full length '
  '(' + '; '.join(f'{n} ' + ' and '.join(f'{v:,}' for v in st[n]['md_len'].values()) + ' frames' for n in SYSTEMS) + '), rollout seed 3000, with the depth scale of the first-half files. MD '
  'reference, floors and the "other MD trajectory" column are the second halves: the floor is one second-half trajectory scored against '
  'the other. "Full model" is the final recipe (squash module trained jointly) scored with squash on; "ablation (full model, squash off)" is the same checkpoints '
  'with the module off (squash_enabled false, squash weight 0); "backbone model" is the same two KL losses trained without the squash module; "comparison model" is the baseline recipe. "mean +/- sd" is over runs; model '
  'curves in the figures pool every rollout of every run (median, 5-95%); pooled MD curves weight the two trajectories equally.\n')
w('Trajectory numbering: 7-PIEZO trajectory 1 is the 16,631-frame run (files 7piezo_R1*), trajectory 2 the 11,103-frame run (7piezo_R2*); 14-PIEZO replica 1 is piezo14_rep01, replica 2 piezo14_rep02.\n')
w('κ = clustering score, the Gaussian KL on the log pair distance of the model against MD, as a share of the same KL for random placement '
  '(0 = MD, 1 = random); "mean part" and "spread part" are its signed parts (mean part > 0: model pair distances longer on average than MD; '
  'spread part > 0: wider spread of log distances); "frames 3000+" scores only rollout frames from 3,000 on.\n')
w('Sources: results/rows_half.jsonl, stats_half.json, forecast_stats_half.json; figures in '
  'figures/.\n')
w('## Calibration of κ on the second halves\n')
w('KL(MD||random) / KL(MD||other MD trajectory) of the log pair distance, per second-half trajectory (the κ floor is its inverse). '
  'Rule, applied per trajectory: a normalized score '
  '(κ or histogram score) is read on a trajectory only when its MD floor is below 1, i.e. the other MD trajectory is closer to MD than random '
  f'placement is; the largest cluster is read only where MD\'s value differs from random placement\'s by at least {BIG_MIN:g}.\n')
w('| system | trajectory | KL(MD||random) | KL(MD||other MD) | ratio | κ floor | histogram-score floor | MD largest cluster | random largest cluster | κ read | histogram score read | largest cluster read |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
ok_k, ok_g, ok_big = {}, {}, {}
for sysname in SYSTEMS:
    gfl = next(r['g_floor'] for r in gr.values() if rows[r['tag']]['system'] == sysname)
    for k, v in st[sysname]['kl_calibration'].items():
        ok_k[(sysname, k)] = 1 / v['calibration'] < 1
        ok_g[(sysname, k)] = gfl[k] < 1
        ok_big[(sysname, k)] = abs(st[sysname]['md_mean_largest_cluster'][k] - st[sysname]['random_mean_largest_cluster']) >= BIG_MIN
        yn = lambda b: 'yes' if b else 'no'
        w(f'| {sysname} | {NAMES[k]} | {v["kl_random"]:.4f} | {v["kl_other_md"]:.4f} | {v["calibration"]:.2f} | {1 / v["calibration"]:.3f} | {gfl[k]:.3f} | '
          f'{st[sysname]["md_mean_largest_cluster"][k]:.2f} | {st[sysname]["random_mean_largest_cluster"]:.2f} | {yn(ok_k[(sysname, k)])} | {yn(ok_g[(sysname, k)])} | {yn(ok_big[(sysname, k)])} |')
w('\nValues on trajectories marked "no" are listed in the tables below, flagged "not read", for completeness and are not to be used as evidence.\n')
nk = lambda n, k: '' if ok_k[(n, k)] else ' (κ not read)'
ng = lambda n, k: '' if ok_g[(n, k)] else ' (not read)'
nb = lambda n, k: '' if ok_big[(n, k)] else ' (not read)'
w('## Definitions (read from the code; file:line)\n')
w('All distances are x,y minimum-image distances in the periodic box (' + at(FO, 'def pair_dist_xy') + '). Per-run values are computed on one run\'s 32 rollouts of one second-half trajectory; "mean +/- sd" is then over runs.\n')
w(f'- κ: Gaussian KL of the model\'s ln pair distance (ln of max(distance, 1) in model units, every 4th frame, all pairs and rollouts pooled) from MD\'s, divided by the same KL for 8,000 uniform random placements (seed 0), so 0 is MD and 1 random ({at(LK, "def gparts")}, {at(LK, "def lnpd")}, {at(LK, "def kl_metrics")}). "Frames 3000+" uses rollout frames from 3,000 on ({at(FH + "roll_heldout.py", "def late_share")}).')
w(f'- histogram score: histogram of all pair distances on 100 bins from 0 to box/sqrt(2), every 4th frame ({at(GRL, "def hist")}; NBIN, STRIDE at {at(GRL, "NBIN, STRIDE")}); score = KL(MD||model) / KL(MD||random placement) of the histograms after merging bins by 1 (7-PIEZO) or 20 (14-PIEZO), the model histogram with +0.5 per bin ({at(GRL, "def kl(")}, {at(GRL, "def score")}, merge at {at(GRL, "MERGE =")}; random: 20,000 placements, seed 0, {at(GRL, "self.Hr")}). The floor is the same score of the other second-half MD trajectory (floor_second.json, read in {at(FH + "score_histogram.py", "floor_second.json")}).')
w(f'- Depth-isolation coupling: Pearson correlation of a protein\'s node depth with its isolation (mean distance to the other proteins in the frame, {at(FO, "def isolation")}), pooled over all proteins, frames 1.. and 32 rollouts of one trajectory ({at(FO, "def coupling")}, applied at {at(FO, "r = coupling(x, node, ref.box)")}). MD: the same pooled correlation over the MD frames that have depth, frame 0 excluded ({at(FO, "self.r[k] = coupling")}).')
w(f'- Node-depth and junction-depth sd ratios: standard deviation of the model\'s node (resp. junction) depth over all rollout frames 1.., proteins (resp. junctions) and rollouts of one trajectory, divided by MD\'s standard deviation over its frames with depth, frame 0 excluded ({at(FO, "node_sd_ratio")}; MD sd at {at(FO, "self.nodeq[k], self.node_sd[k]")}). Depth is in the units of the depth files.')
w(f'- Dynamics KL: for lags 1, 10, 100 and 1,000 frames ({at(LK, "DLAGS =")}) the Gaussian KL of MD\'s change in ln pair distance over the lag from the model\'s, at full frame rate (no striding), pooled over frames, pairs and (model) rollouts; the reported value is the mean of the four lag KLs in nats with no normalizer ({at(LK, "lxf, lmf = ")}, {at(LK, "d = [gparts")}).')
w(f'- Largest cluster: proteins are linked when their distance is at most the contact cutoff (68.93 nm in both systems: {at(FO, "RC = ")} in model units for 7-PIEZO, the 14-PIEZO run configuration\'s fit_rc for 14-PIEZO); the cluster is the connected group and the value is the size of the largest per frame ({at(FO, "def largest_cluster")}), averaged over all frames and rollouts of a trajectory ({at(FO, "big = float(largest_cluster")}); MD likewise over its frames ({at(FO, "self.big[k], self.pdq[k]")}).')
w(f'- MD and random references for the largest cluster (one computation, used in every table and the forecast): MD over all frames of the second half at full frame rate; random placement over 100,000 uniform placements, seed 0 ({at(FO, "self.rand_big")}).')
w(f'- Squash trace ({at(FH + "trace_squash.py", "def trace")}): for each full-model run, {at(FH + "trace_squash.py", "n=8, steps=1000, seed=7")} 8 rollouts of 1,000 steps; start frames are 8 distinct frames drawn at random (numpy generator, seed 7) from the pooled frames of both second-half trajectories ({at(FH + "trace_squash.py", "g.choice")}); initial orientations are independent uniform on [0, 2 pi) per protein and rollout, torch seed 7 ({at(FH + "trace_squash.py", "initial_theta")}). A step is closing when the displacement taken has a positive component towards the nearest neighbour at the start of the step (the module only shortens closing components and keeps their sign, so this is also the proposed step\'s direction), and h is the module\'s contraction factor ({at(FH + "trace_squash.py", "closing = ")}). "Contracted" is a closing step with h < 0.99 and its share is taken over all protein steps (8 x 1,000 x P per run), closing or not; "factor below 0.99 in any direction" is h < 0.99 over the same denominator ({at(FH + "trace_squash.py", "return cnt, hsum")}). Fig. squash (a) shows, per bin of nearest-neighbour distance over the module\'s own hard core (44 bins from 0.8 to 3.0, {at(FH + "trace_squash.py", "EDGES = ")}), the mean h over closing steps only, summing counts and sums over all runs of a system and drawing bins with more than 50 closing steps ({at(FH + "figures.py", "cnt > 50")}); the axis is cut at 1.4 core diameters, beyond which h = 1.')
w(f'- Contact survival: with contact meaning distance at most the cutoff above, the probability that a pair in contact at frame t is still in contact at t+L, pooled over pairs and t, at full frame rate, per rollout; tables and figures show the median and 5-95% over rollouts ({at(FH + "figures.py", "def survival")}).')
w(f'- Overlap: share of pairs closer than the hard core (21.38 nm in both systems; 53.2 model units, {at(FO, "CORE = ")}, for 7-PIEZO), over all pairs, all frames including frame 0 and all rollouts of a trajectory ({at(FO, "\'overlap\': float")}); the per-run value is the mean over the two trajectories. MD: the same share over all its frames ({at(FH + "figures.py", "md_overlap")}).')
w(f'- Calibration ratio: KL(MD||random) / KL(MD||other MD trajectory), the Gaussian KL of the ln pair distance defined for κ, every 4th frame, 8,000 random placements (seed 0), per second-half trajectory ({at(FH + "figures.py", "def kl_table")}).')
w(f'- Forecast: {at(FH + "roll_forecast.py", "NR, STEPS, STRIDE")}: 32 rollouts per trajectory per run (ROLLS overrides), 40,000 steps, every 4th frame kept; runs per system (7-PIEZO all eight, 14-PIEZO all five final-model runs) are listed in the forecast section; windows are 2,000 frames ({at(FH + "forecast.py", "STEPS, STRIDE, WIN")}). Figure curves pool all runs and rollouts; the first model window is compared with MD\'s first 2,000 frames, the last with all MD frames.\n')
MW, ND = {}, {}
VAR = (('sq', 'full model'), ('nosq', 'ablation (full model, squash off)'), ('lk', 'backbone model'), ('base', 'comparison model'))
for sysname in SYSTEMS:
    s = st[sysname]
    R = {v: get(sysname, v) for v, _ in VAR}
    for v in R:
        for r in R[v]:
            r['g'], r['g_floor'] = gr[r['tag']]['g'], gr[r['tag']]['g_floor']
    trajs = list(s['md_len'])
    w(f'\n## {sysname}\n')
    for v, lab in VAR:
        w(f'Checkpoints, {lab}, (run, seed, epoch, epochs trained): {[(r["run"], r["seed"], r["epoch"], r["n_epochs"]) for r in R[v]]}. n = {len(R[v])}.')
    w(f'\nContact cutoff {s["rc_nm"]:.2f} nm, hard core {s["core_nm"]:.2f} nm, box {s["box_nm"]:.0f} nm. Second-half lengths: '
      + ', '.join(f'{NAMES[k]} {s["md_len"][k]} frames' for k in trajs) + '.\n')
    hdr = '| ' + ' | '.join(lab for _, lab in VAR) + ' |'
    sep = '|---|' + '---|' * (len(VAR) + 1)
    w('### Clustering score κ (a training loss)\n')
    for field, title in (('logpair_kl_share', 'κ'), ('late_logpair_kl_share', 'κ, frames 3000+')):
        w(f'\n{title}:\n')
        w('| trajectory ' + hdr + ' other MD trajectory (floor) |')
        w(sep)
        for k in trajs:
            w(f'| {NAMES[k]}{nk(sysname, k)} | ' + ' | '.join(ms([r['kl'][k][field] for r in R[v]]) for v, _ in VAR) + f' | {s["kl_replica"][k][field]:.3f} |')
    w('\nSigned parts (mean part / spread part):\n')
    w('| trajectory ' + hdr + ' other MD trajectory |')
    w(sep)
    for k in trajs:
        rep = s['kl_replica'][k]
        w(f'| {NAMES[k]}{nk(sysname, k)} | ' + ' | '.join(ms([r['kl'][k]['logpair_mean_share'] for r in R[v]], '{:+.3f}') + ' / '
                                        + ms([r['kl'][k]['logpair_spread_share'] for r in R[v]], '{:+.3f}') for v, _ in VAR)
          + f' | {rep["logpair_mean_share"]:+.3f} / {rep["logpair_spread_share"]:+.3f} |')
    for v, lab in (('sq', 'full model'), ('lk', 'backbone model'), ('base', 'comparison model')):
        w(f'\nPer run κ, {lab}, by trajectory: ' + '; '.join(f's{r["seed"]}: ' + ', '.join(f'{r["kl"][k]["logpair_kl_share"]:.3f}' for k in trajs) for r in R[v]))
    w('\n### Measures that are not training losses\n')
    w('Largest cluster, contact survival, histogram score, depth and overlap are not terms of the training loss (the losses are the two KL terms '
      'on the log pair distance and its change, plus the depth and overlap terms of the in-training objective used only for selection).\n')
    w('| measure ' + hdr + ' MD / reference |')
    w('|---|' + '---|' * (len(VAR) + 1))
    for k in trajs:
        t = NAMES[k]
        rows_ = [(f'largest cluster, {t}{nb(sysname, k)}',
                  lambda r, k=k: r['big_' + k], '{:.2f}', f'{s["md_mean_largest_cluster"][k]:.2f} (random {s["random_mean_largest_cluster"]:.2f})'),
                 (f'histogram score, {t}{ng(sysname, k)}', lambda r, k=k: r['g'][k], '{:.3f}', f'floor {R["sq"][0]["g_floor"][k]:.3f}'),
                 (f'depth-isolation coupling, {t}', lambda r, k=k: r['coupling_' + k], '{:.3f}', f'{s["md_depth_coupling"][k]:.3f}'),
                 (f'node-depth sd ratio, {t}', lambda r, k=k: r['node_sd_ratio_' + k], '{:.2f}', '1'),
                 (f'junction-depth sd ratio, {t}', lambda r, k=k: r['hyper_sd_ratio_' + k], '{:.2f}', '1')]
        for name, f, fmt, ref in rows_:
            w(f'| {name} | ' + ' | '.join(ms([f(r) for r in R[v]], fmt) for v, _ in VAR) + f' | {ref} |')
    w('| overlap | ' + ' | '.join(ms([r['overlap'] for r in R[v]], '{:.4f}') for v, _ in VAR) + ' | MD ' + ', '.join(f'{s["md_overlap"][k]:.6f}' for k in trajs) + ' |')
    w('\nMedian node depth (nm): ' + '; '.join(f'{NAMES[k]}: MD {s["md_node_depth_median_nm"][k]:.1f}, ' + ', '.join(
        f'{lab} {s["model_node_depth_median_nm"][v][k]:.1f}' for v, lab in VAR) for k in trajs))
    if sysname == '7-PIEZO':
        NV = (('sq', 'full model', 8), ('lk', 'backbone model', 5), ('base', 'comparison model', 6))
        ND[sysname] = {}
        for k in trajs:
            m = {v: [float(np.median(np.load(os.path.join(ROLLS, f"{r['tag']}_seed{SEED}.npz"))[f'node_depth_{k}'])) * SYSTEMS[sysname]['nm']
                     for r in R[v]] for v, _, _ in NV}
            assert all(len(m[v]) == n for v, _, n in NV), {v: len(m[v]) for v in m}
            ND[sysname][k] = {'n': {v: len(m[v]) for v in m}, 'seed_median_nm': {v: float(np.median(m[v])) for v in m},
                              'p_full_vs_backbone': float(mwu(m['sq'], m['lk'])), 'p_full_vs_comparison': float(mwu(m['sq'], m['base']))}
            print(sysname, k, ND[sysname][k]['n'])
        w('\nPer-run median node depth (nm), median over runs; n = runs; two-sided Mann-Whitney p of the full model against the backbone and comparison models: '
          + '; '.join(f'{NAMES[k]}: ' + ', '.join(f'{lab} {ND[sysname][k]["seed_median_nm"][v]:.1f} (n = {ND[sysname][k]["n"][v]})' for v, lab, _ in NV)
                      + f', p full vs backbone {ND[sysname][k]["p_full_vs_backbone"]:.3f}, p full vs comparison {ND[sysname][k]["p_full_vs_comparison"]:.3f}' for k in trajs))
    w('\nContact survival P(contact at t+L | contact at t), medians over pooled rollouts; MD / ' + ' / '.join(lab for _, lab in VAR) + ':\n')
    for k in trajs:
        md = s['md_survival'][k]
        w(f'- {NAMES[k]}: ' + ', '.join(f'L={L_}: {md[L_]:.2f} / ' + ' / '.join(f'{s["model_survival_median"][v][k][L_]:.2f}' for v, _ in VAR)
                                         for L_ in md if np.isfinite(md[L_])))
    w('\n### Overlap and squash module\n')
    w('Per run overlap without -> with squash: ' + '; '.join(f's{a["seed"]}: {a["overlap"]:.4f} -> {b["overlap"]:.4f}' for a, b in zip(R['nosq'], R['sq'])))
    sqt = list(s['squash_trace'].values())
    w('\nSquash trace (per run, 8 rollouts of 1,000 steps from random second-half MD frames, random initial orientations; distances over the module\'s own hard core): '
      f'share of protein steps the module contracts (a closing step with factor below 0.99) {ms([100 * t["frac_contracted"] for t in sqt], "{:.2f}")} %; '
      f'share with factor below 0.99 in any direction {ms([100 * t["frac"] for t in sqt], "{:.2f}")} %; closing steps {ms([100 * t["frac_closing"] for t in sqt], "{:.1f}")} %.')
    w('\n### Clustering dynamics (a training loss)\n')
    w('| trajectory ' + hdr + ' other MD trajectory (floor) |')
    w(sep)
    for k in trajs:
        w(f'| {NAMES[k]} | ' + ' | '.join(ms([r['kl'][k]['dlogpair_kl'] for r in R[v]]) for v, _ in VAR) + f' | {s["kl_replica"][k]["dlogpair_kl"]:.6f} |')
    metrics = []
    for k in trajs:
        t = NAMES[k]
        metrics += [(f'κ, {t}{nk(sysname, k)}', lambda r, k=k: r['kl'][k]['logpair_kl_share'], '{:.3f}'),
                    (f'κ frames 3000+, {t}{nk(sysname, k)}', lambda r, k=k: r['kl'][k]['late_logpair_kl_share'], '{:.3f}'),
                    (f'dynamics KL, {t}', lambda r, k=k: r['kl'][k]['dlogpair_kl'], '{:.4f}'),
                    (f'largest cluster, {t}{nb(sysname, k)}', lambda r, k=k: r['big_' + k], '{:.2f}'),
                    (f'histogram score, {t}{ng(sysname, k)}', lambda r, k=k: r['g'][k], '{:.3f}'),
                    (f'node-depth sd ratio, {t}', lambda r, k=k: r['node_sd_ratio_' + k], '{:.2f}'),
                    (f'junction-depth sd ratio, {t}', lambda r, k=k: r['hyper_sd_ratio_' + k], '{:.2f}'),
                    (f'depth coupling, {t}', lambda r, k=k: r['coupling_' + k], '{:.3f}')]
    metrics += [('depth gap (pooled)', lambda r: r['depth'], '{:.3f}'), ('coupling gap (pooled)', lambda r: r['coupling_gap'], '{:.3f}'),
                ('overlap', lambda r: r['overlap'], '{:.4f}')]
    cmp_table('### Model vs baseline\n', R['sq'], R['base'], 'full model', 'comparison model', metrics)
    cmp_table('### Model vs same losses without the squash module (isolates the squash module)\n', R['sq'], R['lk'], 'full model', 'backbone model', metrics)
    cmp_table('### Same losses without the squash module vs baseline (isolates the losses)\n', R['lk'], R['base'], 'backbone model', 'comparison model', metrics)
    cmp_table('### Ablation vs comparison model\n', R['nosq'], R['base'], 'ablation', 'comparison model', [m for m in metrics if m[0].split(',')[0] in ('overlap', 'κ', 'dynamics KL', 'largest cluster', 'histogram score')])
    MW[sysname] = {m[0]: {'n_ablation': len(R['nosq']), 'n_comparison': len(R['base']),
                          'p': float(mwu(np.array([m[1](r) for r in R['nosq']]), np.array([m[1](r) for r in R['base']])))}
                   for m in metrics if m[0].split(',')[0] in ('overlap', 'κ', 'dynamics KL', 'largest cluster', 'histogram score')}
    cmp_table('### Squash on vs off (same checkpoints)\n', R['sq'], R['nosq'], 'on', 'off', metrics, paired=True)
    if sysname in fc:
        f = fc[sysname]
        w('\n### Forecast (40,000-frame rollouts of the model with squash, from frame 0 of each second half)\n')
        nr = [r for r in f['runs']]
        w(f'Runs (run, seed, epoch, epochs trained): {nr}. n = {len(nr)}. Plateau = windows 11 to 20 (frames 20,000 to 40,000) of 2,000 frames each; '
          f'rollouts per trajectory: see forecast_stats_half.json / roll_forecast.py (ROLLS). Second-half MD is {", ".join(str(f[k]["md_len"]) for k in trajs)} frames long, so MD windows cover only its own length.\n')
        w('| trajectory | model, largest cluster first window | last window | plateau | MD largest cluster (all frames) | MD windows | κ first window | κ plateau (median of windows) | κ floor |')
        w('|---|---|---|---|---|---|---|---|---|')
        for k in trajs:
            v = f[k]
            w(f'| {NAMES[k]} | {ms(v["model_largest_cluster_first_last_seeds"][0], "{:.2f}")} | {ms(v["model_largest_cluster_first_last_seeds"][1], "{:.2f}")} | '
              f'{ms(v["plateau"]["big"], "{:.2f}")} | {v["md_largest_cluster_all"]:.2f} | {[round(x, 2) for x in v["md_largest_cluster_windows"]]} | '
              f'{ms(v["model_kappa_first_last_seeds"][0])} | {ms(v["plateau"]["kappa"])} | {v["md_kappa_floor"]:.3f} |')
        w(f'\nRandom placement largest cluster {f["random_largest_cluster"]:.2f}. Node depth 5/50/95 % (nm), model first / last window (run mean) against MD all frames and MD first 2,000 frames:')
        for k in trajs:
            v = f[k]
            w(f'- {NAMES[k]}: model {[round(x, 1) for x in v["model_node_depth_q5_50_95_nm_first_last_seedmean"][0]]} / '
              f'{[round(x, 1) for x in v["model_node_depth_q5_50_95_nm_first_last_seedmean"][1]]}, MD all frames {[round(x, 1) for x in v["md_node_depth_q5_50_95_nm_all"]]}, '
              f'MD first 2,000 frames {[round(x, 1) for x in v["md_node_depth_q5_50_95_nm_first2000"]]}')
w('\n## Context: diffusion (MSD ratio model/MD by lag in frames; mean +/- sd over runs)\n')
for sysname in SYSTEMS:
    R = {v: get(sysname, v) for v, _ in VAR}
    w(f'\n{sysname}\n')
    w('| trajectory | variant | ' + ' | '.join(str(g) for g in (1, 10, 100, 1000, 3000)) + ' |')
    w('|---|---|' + '---|' * 5)
    for k in st[sysname]['md_len']:
        for v, lab in VAR:
            w(f'| {NAMES[k]} | {lab} | ' + ' | '.join(ms([r[f'msd_ratio_{k}_{g}'] for r in R[v]], '{:.2f}') for g in (1, 10, 100, 1000, 3000)) + ' |')
STATS = os.path.join(RESULTS, 'stats_half.json')
j = json.load(open(STATS))
j['mwu_ablation_vs_comparison'] = MW
j['node_depth_run_median_mwu'] = ND
json.dump(j, open(STATS, 'w'), indent=1)
open(OUT, 'w').write('\n'.join(L) + '\n')
print(f'wrote {OUT}')
