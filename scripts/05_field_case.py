import os
"""Field case study: one onshore well, 1 m mud-log data (drilling parameters, C1-C5 gas, lithology, formation tops).
No wireline logs and no test results exist for this well, so the outputs are *interpretations*, not validated predictions."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from paths import RESULTS as W, DATA
from paths import PRIVATE
F = os.environ.get('FIELD_XLSX')
if not F:
    raise SystemExit('Set FIELD_XLSX to the mud-log workbook (not included in this repository).')
OUT = W + 'figs/'

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
INK, GREY = '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555', 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'font.family': 'DejaVu Sans'})

# ---------------------------------------------------------------- load
dr = pd.read_excel(F, sheet_name='DRL. DATA', header=0)
d = pd.DataFrame({
    'md': dr.iloc[:, 0], 'rop': dr.iloc[:, 1], 'wob': dr.iloc[:, 2], 'rpm': dr.iloc[:, 6],
    'mw': dr.iloc[:, 14], 'ecd': dr.iloc[:, 16], 'bit': dr.iloc[:, 22], 'mud': dr.iloc[:, 24],
    'tvd': dr.iloc[:, 25], 'lith': dr.iloc[:, 39], 'tg': dr.iloc[:, 44]})
d = d.apply(lambda c: pd.to_numeric(c, errors='coerce') if c.name not in ('mud', 'lith') else c)
gas = pd.read_excel(F, sheet_name='GAS DATA', header=0)
gas = gas.iloc[:, :9]
gas.columns = ['md', 'tg_g', 'c1', 'c2', 'c3', 'ic4', 'nc4', 'ic5', 'nc5']
gas = gas.apply(pd.to_numeric, errors='coerce')
d['md'] = d.md.round()
gas['md'] = gas.md.round()
d = d.merge(gas.drop(columns='tg_g').drop_duplicates('md'), on='md', how='left')

fm = pd.read_excel(F, sheet_name='FM', header=None).iloc[3:, 3:6]
fm.columns = ['name', 'md', 'tvd']
fm = fm[pd.to_numeric(fm.md, errors='coerce').notna()].copy()
fm['md'] = fm.md.astype(float); fm['tvd'] = fm.tvd.astype(float); fm['name'] = fm.name.str.strip()
fm = fm.sort_values('md').reset_index(drop=True)
td = float(fm.md.iloc[-1])
fm = fm[fm.name != 'WELL TD'].reset_index(drop=True)
fm['base'] = list(fm.md.iloc[1:]) + [td]
print(fm)

# ---------------------------------------------------------------- d-exponent
R = d.rop * 3.28084
good = (d.rop > 0.5) & (d.rpm > 20) & (d.wob > 0.5)
dexp = np.log10(R / (60 * d.rpm)) / np.log10(12 * d.wob / (1000 * d.bit))
dexp[~good] = np.nan
d['dxc'] = dexp * (1.03 / d.ecd)
d['dxc_s'] = d.dxc.rolling(21, center=True, min_periods=8).median()

# normal compaction trend: ln(dxc) vs TVD, fitted on claystone only between 150 and 1500 m TVD (binned medians)
sel = d[(d.lith == 'CLST') & d.tvd.between(150, 1500) & d.dxc_s.notna()].copy()
sel['bin'] = (sel.tvd // 50) * 50
bm = sel.groupby('bin').agg(tvd=('tvd', 'median'), dxc=('dxc_s', 'median'), n=('tvd', 'size'))
bm = bm[bm.n >= 10]
slope, icpt = np.polyfit(bm.tvd, np.log(bm.dxc), 1)
d['dxc_n'] = np.exp(icpt + slope * d.tvd)
d['ratio'] = d.dxc_s / d.dxc_n
print('trend slope per km', slope * 1000, 'bins', len(bm))

# OBG: use the contractor's real-time overburden column
obg = pd.to_numeric(dr.iloc[:, 35], errors='coerce').values
d['obg'] = obg
d.loc[d.mud == 'OBM', 'mud_flag'] = 1

# ---------------------------------------------------------------- gas ratios
c = d[['c1', 'c2', 'c3', 'ic4', 'nc4', 'ic5', 'nc5']].fillna(0)
tot = c.sum(axis=1)
d['wh'] = (tot - c.c1) / tot * 100
d['bh'] = (c.c1 + c.c2) / (c.c3 + c.ic4 + c.nc4 + c.ic5 + c.nc5).replace(0, np.nan)
d['ch'] = (c.ic4 + c.nc4 + c.ic5 + c.nc5) / c.c3.replace(0, np.nan)
d['c1c2'] = c.c1 / c.c2.replace(0, np.nan)
d['c1c3'] = c.c1 / c.c3.replace(0, np.nan)
d['show_raw'] = (tot >= 3000) & (c.c2 > 0) & (c.c3 > 0)   # candidate readings: C1-C5 >= 0.3 %, C2 and C3 present
d['show'] = d['show_raw'] & (d['wh'] <= 40)             # Wh > 40 % at high total gas is physically odd (spike / calibration) -> excluded
d['ntg'] = d.tg * (10.0 / d.rop.clip(lower=1))   # ROP-normalised total gas, 10 m/h reference

def haworth(r):
    if not r['show']: return None
    wh, bh = r['wh'], r['bh']
    if wh < 0.5: return 'Dry gas (Wh<0.5)'
    if wh < 17.5: return 'Gas' if (bh > wh) else 'Gas-condensate / wet gas'
    if wh < 40: return 'Oil-leaning (17.5-40)' if bh < wh else 'Wet gas / light oil'
    return 'Heavy / residual (Wh>40)'
d['class'] = d.apply(haworth, axis=1)

# formation summary
rows = []
for _, f in fm.iterrows():
    z = d[(d.md >= f.md) & (d.md < f.base)]
    zs = z[z['show']]
    top = z.lith.value_counts(normalize=True)
    rows.append(dict(formation=f['name'], top_md=f.md, base_md=f.base, n=len(z),
                     dominant_lith=top.index[0] if len(top) else '', tg_max=z.tg.max(), tg_med=z.tg.median(),
                     n_show=len(zs), wh_med=zs.wh.median() if len(zs) else np.nan, bh_med=zs.bh.median() if len(zs) else np.nan,
                     c1c2_med=zs.c1c2.median() if len(zs) else np.nan, ecd_med=z.ecd.median(), mw_med=z.mw.median()))
fsum = pd.DataFrame(rows)
fsum.to_csv(W + 'field_formation_summary.csv', index=False)
print(fsum.round(2).to_string())

stat = dict(
    td_md=td, n_samples=len(d), n_gas_rows=int((d.tg > 0).sum()), n_show_rows=int(d['show'].sum()), n_show_excluded_wh_gt_40=int((d['show_raw'] & ~d['show']).sum()),
    trend_slope_per_km=float(slope * 1000), n_trend_bins=int(len(bm)),
    mw_median=float(d.mw.median()), max_tg=float(d.tg.max()),
    class_counts=d['class'].value_counts().to_dict())
json.dump(stat, open(W + 'field_stats.json', 'w'), indent=1, default=float)
print(json.dumps(stat, indent=1, default=float))
d.to_csv(PRIVATE + 'field_processed.csv', index=False)

# ---------------------------------------------------------------- figures
LC = {'SST': '#f2d16b', 'CLST': '#8d8d89', 'LST': '#7fb6e6', 'SLT': '#c9b79c'}
fig, ax = plt.subplots(1, 7, figsize=(11, 9.5), sharey=True, gridspec_kw={'width_ratios': [0.35, 1, 1, 1.1, 1, 1, 1]})
y = d.md
# 1 lithology
for l, col in LC.items():
    m = (d.lith == l).values
    ax[0].fill_betweenx(y, 0, 1, where=m, color=col, step='mid', lw=0)
ax[0].set_xlim(0, 1); ax[0].set_xticks([]); ax[0].set_title('Mud-log\nlithology', fontsize=8)
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in LC.values()], labels=list(LC.keys()), loc='lower center',
             fontsize=6, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.08))
# 2 ROP
ax[1].plot(d.rop, y, color=BLUE, lw=0.5); ax[1].set_xlim(0, 80); ax[1].set_title('ROP\n(m/h)', fontsize=8)
# 3 dxc
ax[2].plot(d.dxc_s, y, color=ORANGE, lw=0.6, label='dxc (21 m running median)')
ax[2].set_xlim(0.2, 2.2); ax[2].set_title('Corrected\nd-exponent', fontsize=8); 
# 4 pressures (SG)
ax[3].plot(d.ecd, y, color=INK, lw=0.7, label='ECD'); ax[3].plot(d.mw, y, color=BLUE, lw=0.7, ls='--', label='Mud weight in')
ax[3].set_xlim(0.9, 2.2); ax[3].set_title('Mud pressure gradients\n(SG)', fontsize=8); ax[3].legend(fontsize=6, frameon=False, loc='lower left')
# 5 total gas
ax[4].plot(d.tg.clip(lower=0.01), y, color=GREEN, lw=0.6); ax[4].set_xscale('log'); ax[4].set_xlim(0.01, 30); ax[4].set_title('Total gas\n(%, log)', fontsize=8)
# 6 Wh
sh = d[d['show']]
ax[5].scatter(sh.wh, sh.md, s=2, color=MAGENTA, lw=0); ax[5].set_xscale('log'); ax[5].set_xlim(0.1, 100)
for v in (0.5, 17.5, 40): ax[5].axvline(v, color=GREY, lw=0.5, ls=':')
ax[5].set_title('Wetness Wh\n(%)', fontsize=8)
# 7 Bh
ax[6].scatter(sh.bh, sh.md, s=2, color=BLUE, lw=0); ax[6].set_xscale('log'); ax[6].set_xlim(0.5, 1000); ax[6].set_title('Balance Bh', fontsize=8)
for a in ax:
    a.grid(axis='x', lw=0.3, color='#dddddd')
    for _, f in fm.iterrows():
        a.axhline(f.md, color='#bbbbbb', lw=0.4)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
for _, f in fm.iterrows():
    if f.base - f.md >= 100:
        ax[0].text(0.04, f.md + 12, f['name'].split('/')[0].strip()[:18], fontsize=5.2, va='top', ha='left', color=INK)
ax[0].set_ylabel('Measured depth (m)'); ax[0].set_ylim(td, 0)
# mud-type change marker
mt = d[d.mud == 'OBM'].md.min()
for a in ax[1:5]: a.axhline(mt, color=INK, lw=0.6, ls='-.')
ax[1].text(78, mt - 15, 'WBM → OBM', fontsize=6, ha='right', va='bottom')
plt.tight_layout(); plt.savefig(OUT + 'fig_field_composite.png', dpi=300); plt.close()

# Wh-Bh cross-plot
cls_order = ['Gas', 'Gas-condensate / wet gas', 'Oil-leaning (17.5-40)', 'Dry gas (Wh<0.5)']
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
top_f = fsum.sort_values('n_show', ascending=False).head(3)
colors = [BLUE, ORANGE, AQUA]
for (_, f), col in zip(top_f.iterrows(), colors):
    z = sh[(sh.md >= f.top_md) & (sh.md < f.base_md)]
    ax[0].scatter(z.wh, z.bh, s=5, alpha=0.6, color=col, lw=0, label=f"{f.formation.split('/')[0].strip()} (n={len(z)})")
oth = sh[~sh.md.isin(pd.concat([sh[(sh.md >= f.top_md) & (sh.md < f.base_md)].md for _, f in top_f.iterrows()]))]
ax[0].scatter(oth.wh, oth.bh, s=5, alpha=0.5, color=GREY, lw=0, label=f'Other (n={len(oth)})')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('Wetness Wh (%)'); ax[0].set_ylabel('Balance Bh')
for v in (0.5, 17.5, 40): ax[0].axvline(v, color=GREY, lw=0.5, ls=':')
ax[0].plot([0.1, 100], [0.1, 100], color=INK, lw=0.6, ls='--'); ax[0].text(1.3, 1.1, 'Bh = Wh', fontsize=6, rotation=33)
ax[0].legend(fontsize=6, frameon=False, loc='lower left'); ax[0].set_title('(a) Haworth cross-plot', fontsize=8)
# (b) C1/C2 vs C1/C3
z = sh.dropna(subset=['c1c2', 'c1c3'])
ax[1].scatter(z.c1c3, z.c1c2, s=5, alpha=0.5, color=BLUE, lw=0)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('C1/C3'); ax[1].set_ylabel('C1/C2'); ax[1].set_title('(b) Pixler ratios', fontsize=8)
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + 'fig_field_gas_crossplots.png', dpi=300); plt.close()
print('figures written')
