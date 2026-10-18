"""Extra gas diagnostics for the onshore well: ROP-normalised gas, isomer ratios, Bernard parameter, Pixler ordering.
Reads field_processed.csv and field_formation_summary.csv written by field_case.py (private)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import RESULTS as W, DATA, PRIVATE
BLUE, ORANGE, AQUA, YELLOW, INK, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})
d = pd.read_csv(PRIVATE + 'field_processed.csv'); t = pd.read_csv(W + 'field_formation_summary.csv')
d['formation'] = ''
for _, r in t.iterrows():
    d.loc[(d.md >= r.top_md) & (d.md < r.base_md), 'formation'] = r.formation
for c in ['ic4', 'nc4', 'ic5', 'nc5', 'c1', 'c2', 'c3']:
    d[c] = d[c].fillna(0)
d['c4'] = d.ic4 + d.nc4; d['c5'] = d.ic5 + d.nc5
rop_ref = float(d.loc[d['show'], 'rop'].median())
d['tg_norm'] = d.tg * rop_ref / d.rop.clip(lower=0.5)               # gas scaled to a common penetration rate
d['ic4nc4'] = d.ic4 / d.nc4.replace(0, np.nan); d['ic5nc5'] = d.ic5 / d.nc5.replace(0, np.nan)
d['c2c3'] = d.c2 / d.c3.replace(0, np.nan); d['bern'] = d.c1 / (d.c2 + d.c3)
r2 = d.c1 / d.c2.replace(0, np.nan); r3 = d.c1 / d.c3.replace(0, np.nan); r4 = d.c1 / d.c4.replace(0, np.nan); r5 = d.c1 / d.c5.replace(0, np.nan)
d['pixler_ok'] = (r2 < r3) & (r3 < r4) & (r4 < r5)
d['pixler_ok_345'] = (r2 < r3) & (r3 < r4)
s = d[d['show']].copy()
forms = [f for f in ['Lockhart', 'Hangu', 'Lumshiwal', 'Chichali'] if (s.formation == f).any()]
rows = []
for f in forms:
    z = s[s.formation == f]
    c5ok = z.c5 > 0
    rows.append(dict(formation=f, n=len(z), tgn_med=z.tg_norm.median(), ic4nc4=z.ic4nc4.median(), ic5nc5=z.ic5nc5.median(),
                     c2c3=z.c2c3.median(), bern=z.bern.median(), pix_ok=100 * z.pixler_ok.mean()))
tab = pd.DataFrame(rows); tab.to_csv(W + 'field_gas_diag.csv', index=False); print(tab.round(2).to_string())
stats = dict(rop_ref=rop_ref, bern_all_med=float(s.bern.median()), bern_min=float(s.bern.min()), bern_max=float(s.bern.max()),
             ic4nc4_gt1_pct=float(100 * (s.ic4nc4 > 1).mean()), c2c3_gt1_pct=float(100 * (s.c2c3 > 1).mean()),
             pix_ok_345_pct=float(100 * s.pixler_ok_345.mean()), pix_ok_all_pct=float(100 * s.pixler_ok.mean()),
             tgn_max_over_tg_max=float(s.tg_norm.max() / s.tg.max()), n=int(len(s)),
             corr_tg_rop=float(s[['tg', 'rop']].corr(method='spearman').iloc[0, 1]), corr_tgn_rop=float(s[['tg_norm', 'rop']].corr(method='spearman').iloc[0, 1]))
json.dump(stats, open(W + 'field_gas_diag.json', 'w'), indent=1); print(stats)

fig, ax = plt.subplots(1, 4, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.2, 1, 1, 1]})
sec = d[(d.md >= 3480) & (d.md <= d.md.max())]
ax[0].plot(sec.tg, sec.md, color=GREY, lw=0.7, label='Total gas'); ax[0].plot(sec.tg_norm, sec.md, color=BLUE, lw=0.7, label=f'ROP-normalised (to {rop_ref:.0f} m/h)')
ax[0].set_xscale('log'); ax[0].set_xlim(0.03, 50); ax[0].invert_yaxis(); ax[0].set_ylabel('Measured depth (m)'); ax[0].set_xlabel('Gas (%)'); ax[0].legend(frameon=False, fontsize=6.5, loc='lower left')
ax[0].set_title('(a) Raw and normalised gas', fontsize=8.5)
cols = dict(zip(forms, [BLUE, ORANGE, AQUA, YELLOW]))
for k, (col, lab, ttl) in enumerate([('ic4nc4', 'iC4 / nC4', '(b) Butane isomers'), ('ic5nc5', 'iC5 / nC5', '(c) Pentane isomers'), ('bern', 'C1 / (C2 + C3)', '(d) Bernard parameter')], start=1):
    data = [s.loc[s.formation == f, col].dropna().values for f in forms]
    bp = ax[k].boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for patch, f in zip(bp['boxes'], forms): patch.set(facecolor=cols[f], alpha=0.55, lw=0.6)
    ax[k].set_xticks(range(1, len(forms) + 1)); ax[k].set_xticklabels([f[:5] + '.' if len(f) > 6 else f for f in forms], fontsize=7); ax[k].set_ylabel(lab); ax[k].set_title(ttl, fontsize=8.5)
ax[3].axhline(100, color=INK, lw=0.6, ls=':'); ax[3].text(0.6, 104, 'thermogenic below ~100', fontsize=6.5); ax[3].set_yscale('log'); ax[3].set_ylim(1, 1500)
for a in ax:
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(W + 'figs/fig_field_gas_diag.png', dpi=300); plt.close()
