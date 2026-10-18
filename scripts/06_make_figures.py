import os
"""Figures for the manuscript, built only from saved experiment outputs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import r2_score

from paths import RESULTS as W, DATA
O = W + 'figs/'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
INK, GREY = '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555', 'font.family': 'DejaVu Sans'})
names = ['Sandstone', 'Sand/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']

# ------------------------------------------------ Fig 1: workflow diagram
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
cols = [('1  Data', ['Wireline logs (GR, RHOB,', 'NPHI, DTC, DTS, Res.)', 'Drilling data (ROP, WOB,', 'RPM, mud weight)', 'Mud gas C1-C5']),
        ('2  Quality control', ['Calliper / DRHO check', 'Physical range limits', 'Well-wise train / test split', '(no well in both sets)']),
        ('3  Interpretation', ['Lithology classifier', 'Density and shear', 'slowness reconstruction', 'Overburden, Eaton PPG,', 'Eaton FG, gas ratios']),
        ('4  Outputs', ['Composite log', 'Pressure window vs', 'mud weight used', 'Gas-character flags']),
        ('5  Use', ['Review by wellsite', 'geologist / engineer', 'Recalibrate with LOT,', 'RFT, core when available'])]
xs = [1, 21, 41, 61, 81]
for (t, lines), x in zip(cols, xs):
    ax.add_patch(FancyBboxPatch((x, 3), 17.5, 33, boxstyle='round,pad=0.2,rounding_size=1.2', fc='#f4f6f8', ec='#9aa5b1', lw=0.8))
    ax.text(x + 8.75, 33, t, ha='center', va='center', fontsize=7.8, weight='bold', color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 8.75, 27.5 - i * 3.9, l, ha='center', va='center', fontsize=6.1, color=INK)
for x in xs[:-1]:
    ax.add_patch(FancyArrowPatch((x + 18, 19), (x + 20.4, 19), arrowstyle='-|>', mutation_scale=9, color=INK, lw=0.8))
plt.savefig(O + 'fig1_workflow.png', dpi=300, bbox_inches='tight'); plt.close()

# ------------------------------------------------ Fig 2: lithology
m = json.load(open(W + 'litho_metrics.json'))
pc = json.load(open(W + 'litho_perclass.json'))
cm = np.load(W + 'litho_cm_best.npy').astype(float)
order = ['Empirical thresholds', 'Random Forest', 'XGBoost', 'SVM (RBF)']
fig, ax = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={'width_ratios': [1, 1.25]})
x = np.arange(len(order)); wdt = 0.26
for k, (key, lab, col) in enumerate([('accuracy', 'Accuracy', BLUE), ('balanced_acc', 'Balanced accuracy', ORANGE), ('macro_f1', 'Macro F1', AQUA)]):
    v = [m[o][key] for o in order]
    b = ax[0].bar(x + (k - 1) * wdt, v, wdt - 0.03, color=col, label=lab)
    for xx, vv in zip(x + (k - 1) * wdt, v): ax[0].text(xx, vv + 0.01, f'{vv:.2f}', ha='center', fontsize=6)
ax[0].set_xticks(x); ax[0].set_xticklabels(['Threshold\nrules', 'Random\nForest', 'XGBoost', 'SVM\n(RBF)']); ax[0].set_ylim(0, 1)
ax[0].set_ylabel('Score (5-fold, wells held out)'); ax[0].legend(frameon=False, fontsize=7, loc='upper left', ncol=1)
ax[0].set_title('(a) Model comparison', fontsize=8.5)
cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
im = ax[1].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
ax[1].set_xticks(range(12)); ax[1].set_yticks(range(12)); ax[1].set_xticklabels(names, rotation=60, ha='right', fontsize=6.5); ax[1].set_yticklabels(names, fontsize=6.5)
for i in range(12):
    for j in range(12):
        if cmn[i, j] >= 0.05: ax[1].text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='white' if cmn[i, j] > 0.5 else INK)
ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Expert label'); ax[1].set_title(f"(b) Row-normalised confusion matrix, {pc['best']}", fontsize=8.5)
plt.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02)
plt.tight_layout(); plt.savefig(O + 'fig2_lithology.png', dpi=300); plt.close()

# ------------------------------------------------ Fig 3: reconstruction scatter
rm = json.load(open(W + 'recon_metrics.json'))
rho = np.load(W + 'recon_RHOB.npz'); dts = np.load(W + 'recon_DTS.npz'); nu = np.load(W + 'recon_nu.npz')
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
def hexp(a, x, y, lo, hi, xl, title, r2):
    hb = a.hexbin(x, y, gridsize=70, extent=(lo, hi, lo, hi), mincnt=1, bins='log', cmap='Blues', linewidths=0)
    a.plot([lo, hi], [lo, hi], color=INK, lw=0.7, ls='--'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel(f'Measured {xl}'); a.set_ylabel(f'Predicted {xl}'); a.set_title(title, fontsize=8.5)
    a.text(0.04, 0.93, f'R² = {r2:.2f}', transform=a.transAxes, fontsize=7.5, va='top')
def best(tag):
    o = rm[tag][0]
    return max(['MLP', 'XGBoost'], key=lambda k: o[k]['r2'])
bR, bD = best('RHOB'), best('DTS')
bN = max(['MLP', 'XGBoost'], key=lambda k: rm['nu'][k]['r2'])
hexp(ax[0], rho['y'], rho[bR], 1.6, 3.1, 'density (g/cm³)', f'(a) Bulk density, {bR}', rm['RHOB'][0][bR]['r2'])
hexp(ax[1], dts['y'], dts[bD], 80, 400, 'DTS (µs/ft)', f'(b) Shear slowness, {bD}', rm['DTS'][0][bD]['r2'])
ok = np.isfinite(nu['nu_true']) & np.isfinite(nu[bN]) & (nu['nu_true'] > 0) & (nu['nu_true'] < 0.5)
hexp(ax[2], nu['nu_true'][ok], nu[bN][ok], 0.05, 0.45, "ν", f"(c) ν from predicted DTS, {bN}", rm['nu'][bN]['r2'])
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(O + 'fig3_reconstruction.png', dpi=300); plt.close()

# ------------------------------------------------ Fig 4: Bi-LSTM
try:
    bm = json.load(open(W + 'bilstm_metrics.json')); z = np.load(W + 'bilstm_fold0.npz')
    fig, ax = plt.subplots(1, 4, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.1]})
    rng = np.random.RandomState(3); idx = rng.choice(len(z['y']), 3, replace=False)
    for k, i in enumerate(idx):
        depth = np.arange(z['y'].shape[1]) * 0.15
        a = ax[k]; a.plot(z['y'][i], depth, color=INK, lw=0.9, label='Measured'); a.plot(z['lstm'][i], depth, color=ORANGE, lw=0.9, label='Bi-LSTM')
        a.plot(z['xgb'][i], depth, color=BLUE, lw=0.7, label='XGBoost'); a.set_ylim(depth.max(), 0); a.set_xlabel('DTC (µs/ft)')
        a.set_title(f'({"abc"[k]}) Held-out well, window {k + 1}', fontsize=8)
        if k == 0: a.set_ylabel('Relative depth (m)'); a.legend(frameon=False, fontsize=6.5, loc='lower left')
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    keys = ['Gardner-inverse baseline', 'XGBoost (point-wise)', 'BiLSTM']; labs = ['Gardner\ninverse', 'XGBoost', 'Bi-LSTM']
    r = [bm[k]['r2'] for k in keys]; ax[3].bar(range(3), r, color=[GREY, BLUE, ORANGE], width=0.55)
    for i, v in enumerate(r): ax[3].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=7)
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels(labs); ax[3].set_ylabel('R² (3-fold, wells held out)'); ax[3].set_title('(d) Mean R²', fontsize=8)
    ax[3].set_ylim(min(0, min(r) - 0.1), 1)
    for s in ('top', 'right'): ax[3].spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(O + 'fig4_bilstm.png', dpi=300); plt.close()
except FileNotFoundError:
    print('bilstm not ready')
print('ok')
