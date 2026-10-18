"""Two diagnostic figures built from saved results: where the lithology models fail, and per-well geomechanics."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from paths import RESULTS as W, DATA

BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a86', '#222222'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})
codes = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}
names = ['Sandstone', 'Sand/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']

df = pd.read_csv(DATA + 'train.csv', sep=';', usecols=['WELL', 'FORCE_2020_LITHOFACIES_LITHOLOGY'])
y = df.FORCE_2020_LITHOFACIES_LITHOLOGY.map(codes).values
g = df.WELL.values
rf, xg = np.load(W + 'litho_pred_rf.npy'), np.load(W + 'litho_pred_xgb.npy')
f1 = f1_score(y, rf, average=None, labels=range(12))
counts = np.array([(y == i).sum() for i in range(12)])
imp = json.load(open(W + 'litho_featimp.json'))

# ---- figure A: failure analysis
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1.15, 1, 1]})
ax[0].scatter(counts, f1, s=22, color=BLUE, zorder=3)
for i, n in enumerate(names):
    ax[0].annotate(n, (counts[i], f1[i]), xytext=(4, 3), textcoords='offset points', fontsize=6.5)
ax[0].set_xscale('log'); ax[0].set_xlabel('Samples in class (log scale)'); ax[0].set_ylabel('F1, Random Forest'); ax[0].set_ylim(-0.05, 0.95)
ax[0].set_title('(a) Class size and F1', fontsize=8.5)
wells = np.unique(g)
acc = lambda p: np.array([(p[g == w] == y[g == w]).mean() for w in wells])
bins = np.linspace(0, 1, 21)
ax[1].hist(acc(rf), bins=bins, color=BLUE, alpha=0.8, label='Random Forest')
ax[1].hist(acc(xg), bins=bins, color=ORANGE, alpha=0.55, label='XGBoost')
ax[1].set_xlabel('Accuracy in one held-out well'); ax[1].set_ylabel('Number of wells'); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title('(b) Spread between wells', fontsize=8.5)
order = sorted(imp, key=imp.get)
ax[2].barh(order, [imp[k] for k in order], color=AQUA, height=0.6)
ax[2].set_xlabel('Importance (refit on all wells)'); ax[2].set_title('(c) Feature importance', fontsize=8.5)
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(W + 'figs/fig_litho_diagnostics.png', dpi=300); plt.close()

# ---- figure B: per-well geomechanics
t = pd.read_csv(W + 'geomech_perwell.csv')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
col = [BLUE if v >= 0 else ORANGE for v in t.rho_r2]
ax[0].bar(t.well, t.rho_r2, color=col, width=0.6); ax[0].axhline(0, color=INK, lw=0.6)
ax[0].set_ylabel('R² of density, well held out'); ax[0].set_title('(a) Density reconstruction by well', fontsize=8.5)
ax[0].tick_params(axis='x', rotation=45)
ok = t[t.trend_ok]
x = np.arange(len(ok)); b = ok.mw_below_pp.values; m = ok.mw_between.values; a_ = ok.mw_above_fg.values
ax[1].bar(x, b, color=ORANGE, width=0.55, label='Below Eaton PP'); ax[1].bar(x, m, bottom=b, color=AQUA, width=0.55, label='Inside window')
ax[1].bar(x, a_, bottom=b + m, color=GREY, width=0.55, label='Above Eaton FG')
ax[1].set_xticks(x); ax[1].set_xticklabels(ok.well); ax[1].set_ylabel('Share of mud-weight samples (%)')
ax[1].set_title('(b) Mud weight against the estimated window', fontsize=8.5); ax[1].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(W + 'figs/fig_geomech_by_well.png', dpi=300); plt.close()
print('ok')
