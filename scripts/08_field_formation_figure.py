"""Gas-ratio distribution by formation for the onshore well.
Reads private/field_processed.csv and results/field_formation_summary.csv, both written by 05_field_case.py.
Needs the proprietary workbook, so the figure is not part of the public results folder."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import RESULTS as W, PRIVATE

BLUE, ORANGE, AQUA, YELLOW, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#222222'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})

d = pd.read_csv(PRIVATE + 'field_processed.csv')
tops = pd.read_csv(W + 'field_formation_summary.csv')
d['formation'] = ''
for _, r in tops.iterrows():
    d.loc[(d.md >= r.top_md) & (d.md < r.base_md), 'formation'] = r.formation
d['c4'] = d.ic4.fillna(0) + d.nc4.fillna(0)
d['c1c4'] = d.c1 / d.c4.replace(0, np.nan)
s = d[d['show']].copy()
forms = [f for f in ['Lockhart', 'Hangu', 'Lumshiwal', 'Chichali'] if (s.formation == f).any()]
cols = dict(zip(forms, [BLUE, ORANGE, AQUA, YELLOW]))

fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for k, (col, lab) in enumerate([('wh', 'Wetness Wh (%)'), ('bh', 'Balance Bh')]):
    data = [s.loc[s.formation == f, col].dropna().values for f in forms]
    bp = ax[k].boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for patch, f in zip(bp['boxes'], forms):
        patch.set(facecolor=cols[f], alpha=0.55, lw=0.6)
    rng = np.random.RandomState(1)
    for i, v in enumerate(data):
        ax[k].scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=INK, alpha=0.35, lw=0)
    ax[k].set_xticks(range(1, len(forms) + 1)); ax[k].set_xticklabels([f'{f}\n(n={len(v)})' for f, v in zip(forms, data)])
    ax[k].set_ylabel(lab)
ax[0].axhline(17.5, color=INK, lw=0.6, ls=':'); ax[0].text(0.55, 17.9, 'gas / wet-gas boundary 17.5%', fontsize=6.5)
ax[0].set_title('(a) Wetness by formation', fontsize=8.5); ax[1].set_title('(b) Balance by formation', fontsize=8.5); ax[1].set_yscale('log')
for f in forms:
    z = s[(s.formation == f)].dropna(subset=['c1c2', 'c1c4'])
    ax[2].scatter(z.c1c4, z.c1c2, s=7, alpha=0.6, color=cols[f], lw=0, label=f)
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('C1/C4'); ax[2].set_ylabel('C1/C2')
ax[2].set_title('(c) Pixler ratios by formation', fontsize=8.5); ax[2].legend(frameon=False, fontsize=6.5, loc='upper left')
for a in ax:
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(W + 'figs/fig_field_by_formation.png', dpi=300); plt.close()
print('ok', {f: int((s.formation == f).sum()) for f in forms})
