"""Fifth-review analyses (JHIR reviewer): calibration of composed values, edge-cut sensitivity incl. 0.005,
initial vs revised membership functions on RQ3. Uses only saved relation matrices in results2/ and
results2_pre_expert_review/. Writes results2/revision5.json and fig3_calibration.{png,tiff}."""
import json, sys, itertools
import numpy as np
sys.path.insert(0, '.')
import rq3x as X
from scipy.stats import spearmanr

R2 = 'results2'
ps = json.load(open(f'{R2}/paper_stats.json'))
order = list(ps['flow']['development_2013_2018']['weighted_mean_activation'].keys())
prev = np.array([ps['flow']['development_2013_2018']['weighted_mean_activation'][c] for c in order])
prev_pre = np.array([ps['pre_review']['flow']['development_2013_2018']['weighted_mean_activation'][c] for c in order])
Ed, Eh = np.load(f'{R2}/E_dev.npy'), np.load(f'{R2}/E_hold.npy')
n = 24
OFF = ~np.eye(n, dtype=bool)
T5 = ['product', 'hamacher', 'yager2', 'lukasiewicz', 'min']
OUT = {}


def calib(pred, obs, nb=10):
    pred, obs = np.asarray(pred), np.asarray(obs)
    o = np.argsort(pred, kind='stable')
    bins = np.array_split(o, nb)
    ece = sum(len(b) * abs(pred[b].mean() - obs[b].mean()) for b in bins) / len(pred)
    A = np.vstack([np.ones_like(pred), pred]).T
    a, b = np.linalg.lstsq(A, obs, rcond=None)[0]
    return {'bias': float((pred - obs).mean()), 'mae': float(np.abs(pred - obs).mean()),
            'ece': float(ece), 'slope': float(b), 'intercept': float(a),
            'rho': float(spearmanr(pred, obs).statistic),
            'bins': [[float(pred[b_].mean()), float(obs[b_].mean())] for b_ in bins]}


def recal(pred_dev, edge_dev, pred_hold):
    """Linear recalibration fitted within development (dev paths -> dev edges), applied to held-out predictions."""
    A = np.vstack([np.ones_like(pred_dev), pred_dev]).T
    a, b = np.linalg.lstsq(A, edge_dev, rcond=None)[0]
    return np.clip(a + b * pred_hold, 0, 1), (float(a), float(b))


def all_preds(E, pv):
    P = {t: X.paths(E, X.T_FIXED[t]) for t in T5}
    P['reverse_odds'], _ = X.reverse_baseline(E, pv)
    P['rank3'] = X.lowrank_baseline(E, 3, dyad=True)
    return P


# ---------- A. calibration ----------
P = all_preds(Ed, prev)
cal = {}
for k, v in P.items():
    cal[k] = calib(v[OFF], Eh[OFF])
for t in ['product', 'hamacher', 'min']:
    rc, coef = recal(P[t][OFF], Ed[OFF], P[t][OFF])
    c = calib(rc, Eh[OFF]); c['coef'] = coef
    cal[t + '_recal'] = c
OUT['calibration'] = cal

# node jackknife of calibration metrics and differences
keys = ['product', 'hamacher', 'min', 'product_recal', 'hamacher_recal']
jk = {k: {m: [] for m in ['mae', 'ece', 'bias', 'slope', 'rho']} for k in keys}
for c in range(n):
    keep = [i for i in range(n) if i != c]
    E1, H1 = Ed[np.ix_(keep, keep)], Eh[np.ix_(keep, keep)]
    off = ~np.eye(n - 1, dtype=bool)
    for t in ['product', 'hamacher', 'min']:
        p = X.paths(E1, X.T_FIXED[t])[off]
        r = calib(p, H1[off])
        for m in jk[t]: jk[t][m].append(r[m])
        if t != 'min':
            rc, _ = recal(p, E1[off], p)
            r = calib(rc, H1[off])
            for m in jk[t + '_recal']: jk[t + '_recal'][m].append(r[m])


def jse(x):
    x = np.asarray(x); m = len(x)
    return float(np.sqrt((m - 1) / m * ((x - x.mean()) ** 2).sum()))


diffs = {}
for m in ['mae', 'ece']:
    for a, b in [('hamacher', 'product'), ('product_recal', 'hamacher'), ('product_recal', 'product')]:
        d = np.array(jk[a][m]) - np.array(jk[b][m])
        full = cal[a][m] - cal[b][m]
        diffs[f'{m}:{a}-{b}'] = {'point': float(full), 'jk_se': jse(d), 'share_negative': float((d < 0).mean())}
OUT['calibration_jackknife'] = diffs
OUT['calibration_jackknife_range'] = {k: {m: [float(min(v[m])), float(max(v[m]))] for m in v} for k, v in jk.items()}

# ---------- B. edge-cut sensitivity ----------
cut = {}
for eps in [0.0, 0.005, 0.01, 0.05]:
    E = np.where(Ed >= eps, Ed, 0.0) if eps > 0 else Ed
    Pe = all_preds(E, prev)
    r = {k: float(spearmanr(v[OFF], Eh[OFF]).statistic) for k, v in Pe.items()}
    r['n_zero_reverse'] = int((OFF & (E.T == 0)).sum())
    r['best_tnorm'] = max(T5, key=lambda t: r[t] if np.isfinite(r[t]) else -9)
    cut[str(eps)] = r
OUT['edge_cut'] = cut

# ---------- C. initial vs revised membership functions ----------

PR = 'results2_pre_expert_review'
Edp, Ehp = np.load(f'{PR}/E_dev.npy'), np.load(f'{PR}/E_hold.npy')
Pp = all_preds(Edp, prev_pre)
pre = {k: float(spearmanr(v[OFF], Ehp[OFF]).statistic) for k, v in Pp.items()}
pre['n_edges'] = int((OFF & (Edp >= 0.01)).sum())
pre['mae_product'] = float(np.abs(Pp['product'] - Ehp)[OFF].mean())
pre['mae_hamacher'] = float(np.abs(Pp['hamacher'] - Ehp)[OFF].mean())
pre['rho_dev_initial_vs_revised'] = float(spearmanr(Edp[OFF], Ed[OFF]).statistic)
for f in ['rq1', 'rq2', 'rq3']:
    pre[f + '_file'] = json.load(open(f'{PR}/{f}.json'))
OUT['initial_functions'] = pre
json.dump(OUT, open('results2/revision5.json', 'w'), indent=1)

# ---------- D. calibration figure (grayscale) ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, ax = plt.subplots(figsize=(3.4, 3.3), dpi=600)
style = {'product': ('o', '-', '0.0', 'Product'), 'hamacher': ('s', '--', '0.35', 'Hamacher'),
         'min': ('^', ':', '0.55', 'Minimum'), 'product_recal': ('D', '-.', '0.15', 'Product, recalibrated')}
for k, (mk, ls, col, lab) in style.items():
    b = np.array(cal[k]['bins'])
    ax.plot(b[:, 0], b[:, 1], marker=mk, linestyle=ls, color=col, markersize=4, linewidth=1,
            markerfacecolor='white' if k in ('hamacher', 'product_recal') else col, label=lab)
lim = 1.05 * max(max(max(x) for x in cal[k]['bins']) for k in style)
ax.plot([0, lim], [0, lim], color='0.7', linewidth=0.8, zorder=0)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('Mean composed value (decile of pairs)')
ax.set_ylabel('Mean held-out relation')
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout()
fig.savefig('fig3_calibration.png'); fig.savefig('fig3_calibration.tiff', pil_kwargs={'compression': 'tiff_lzw'})
print('done')
