import os
"""Geomechanics chain on held-out FORCE 2020 wells.
For every eligible well the density and shear-slowness models are trained on the other wells only. Overburden, Eaton
pore pressure and Eaton fracture gradient are then computed with the corrected equations. The open data contain no
measured pore pressure or leak-off tests, so the check is plausibility against the mud weight actually used.
Eligibility (fixed before looking at results): at least 95 % of the well's mud-weight values lie in 0.9-2.6 SG
(the column mixes units across wells) and at least 1000 samples have GR, RHOB, DTC, NPHI and mud weight."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import theilslopes, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

from paths import RESULTS as W, DATA
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
INK, GREY = '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555', 'font.family': 'DejaVu Sans'})

df = pd.read_csv(DATA + 'train.csv', sep=';')
df['LRDEP'] = np.log10(df.RDEP.clip(lower=0.01))
codes = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}
df['y'] = df.FORCE_2020_LITHOFACIES_LITHOLOGY.map(codes)
df['rf'] = np.load(W + 'litho_pred_rf.npy')          # leave-wells-out predictions from the lithology experiment

mw_ok = df.groupby('WELL').MUDWEIGHT.apply(lambda s: s.dropna().between(0.9, 2.6).mean() if s.notna().any() else 0)
cand = mw_ok[mw_ok > 0.95].index
cnt = df[df.WELL.isin(cand)].dropna(subset=['GR', 'RHOB', 'DTC', 'NPHI', 'MUDWEIGHT']).groupby('WELL').size()
wells = list(cnt[cnt >= 1000].sort_values(ascending=False).index)
print('eligible wells:', wells, flush=True)

PN = 1.03
f_rho = ['GR', 'NPHI', 'DTC', 'LRDEP', 'DEPTH_MD', 'CALI']
f_dts = ['GR', 'RHOB_use', 'NPHI', 'DTC', 'LRDEP', 'DEPTH_MD']
XGB = dict(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.7, tree_method='hist', n_jobs=2, random_state=0)


def nu_of(dts, dtc):
    r2 = (dts / dtc) ** 2
    out = (0.5 * r2 - 1) / (r2 - 1)
    return np.where((out > 0) & (out < 0.5), out, np.nan)


def obg_from(z, rho):
    rho = pd.Series(rho).interpolate(limit_direction='both').values
    dz = np.diff(z, prepend=z[0])
    return (2.0 * z[0] + np.cumsum(rho * dz)) / z       # unlogged top assumed 2.0 g/cm3, water column ignored


def run_well(well):
    w = df[df.WELL == well].sort_values('DEPTH_MD').copy()
    tr = df[df.WELL != well].copy()
    q = tr[(tr.RHOB.between(1.5, 3.2)) & (tr.DTC.between(40, 200)) & tr[['GR', 'NPHI']].notna().all(axis=1)]
    s = q.sample(min(200000, len(q)), random_state=0)
    m_rho = xgb.XGBRegressor(**XGB).fit(s[f_rho].values, s.RHOB.values)
    w['RHOB_ml'] = np.nan
    ok = w[['GR', 'NPHI', 'DTC']].notna().all(axis=1)
    w.loc[ok, 'RHOB_ml'] = m_rho.predict(w.loc[ok, f_rho].values)
    w['RHOB_use'] = w.RHOB.where(w.RHOB.between(1.5, 3.2)).fillna(w.RHOB_ml)
    t2 = tr.copy(); t2['RHOB_use'] = t2.RHOB
    q = t2.dropna(subset=['GR', 'RHOB_use', 'NPHI', 'DTC', 'DTS'])
    q = q[q.DTS.between(70, 500) & q.DTC.between(40, 200) & q.RHOB_use.between(1.5, 3.2)]
    s = q.sample(min(200000, len(q)), random_state=0)
    m_dts = xgb.XGBRegressor(**XGB).fit(s[f_dts].values, s.DTS.values)
    w['DTS_ml'] = np.nan
    ok2 = w[['GR', 'RHOB_use', 'NPHI', 'DTC']].notna().all(axis=1)
    w.loc[ok2, 'DTS_ml'] = m_dts.predict(w.loc[ok2, f_dts].values)
    w['nu_ml'] = nu_of(w.DTS_ml, w.DTC)
    w['nu_meas'] = nu_of(w.DTS.where(w.DTS.between(70, 500)), w.DTC)
    z = w.DEPTH_MD.values
    has_rho = w.RHOB.between(1.5, 3.2).any()
    w['OBG_meas'] = obg_from(z, w.RHOB.where(w.RHOB.between(1.5, 3.2)).values) if has_rho else np.nan
    w['OBG_ml'] = obg_from(z, w.RHOB_ml.values)
    w['OBG'] = w.OBG_meas if has_rho else w.OBG_ml
    # normal sonic trend on shales (GR > 75) in the shallowest 40 % of the logged interval, robust fit
    sh = w[(w.GR > 75) & w.DTC.between(40, 200)]
    lim = z.min() + 0.40 * (z.max() - z.min())
    s0 = sh[sh.DEPTH_MD <= lim]
    trend_ok, b, a = False, np.nan, np.nan
    if len(s0) > 300:
        ss = s0.sample(min(2000, len(s0)), random_state=0)
        b, a, _, _ = theilslopes(np.log(ss.DTC.values), ss.DEPTH_MD.values)
        rho_s, p_s = spearmanr(ss.DEPTH_MD, ss.DTC)
        trend_ok = bool((b < 0) and (p_s < 0.05))
    w['DTC_n'] = np.exp(a + b * z) if trend_ok else np.nan
    w['PPG'] = np.maximum(w.OBG - (w.OBG - PN) * (w.DTC_n / w.DTC) ** 3, PN)
    w.loc[~(w.GR > 75), 'PPG'] = np.nan                   # pore pressure only in shale, where the compaction response holds
    w['FG_ml'] = w.PPG + w.nu_ml / (1 - w.nu_ml) * (w.OBG_ml - w.PPG)
    w['FG_meas'] = w.PPG + w.nu_meas / (1 - w.nu_meas) * (w.OBG_meas - w.PPG)

    def sc(f, a_, b_):
        m = a_.notna() & b_.notna()
        return (float(f(a_[m], b_[m])) if m.sum() > 50 else np.nan), int(m.sum())
    rho_meas = w.RHOB.where(w.RHOB.between(1.5, 3.2)); dts_meas = w.DTS.where(w.DTS.between(70, 500))
    r = dict(well=well, n=int(len(w)), top=float(z.min()), base=float(z.max()))
    r['rho_r2'], r['rho_n'] = sc(r2_score, rho_meas, w.RHOB_ml)
    r['rho_mae'], _ = sc(mean_absolute_error, rho_meas, w.RHOB_ml)
    r['dts_r2'], r['dts_n'] = sc(r2_score, dts_meas, w.DTS_ml)
    r['dts_mae'], _ = sc(mean_absolute_error, dts_meas, w.DTS_ml)
    r['nu_mae'], _ = sc(mean_absolute_error, w.nu_meas, w.nu_ml)
    r['obg_diff_sg'] = float(np.nanmean(np.abs(w.OBG_meas - w.OBG_ml))) if has_rho else np.nan
    r['trend_ok'] = trend_ok; r['trend_b_per_km'] = float(b * 1000) if not np.isnan(b) else np.nan
    mm = w.MUDWEIGHT.notna() & w.PPG.notna() & w.FG_ml.notna()
    r['n_mw'] = int(mm.sum())
    if mm.sum() > 50:
        r['mw_between'] = float(((w.MUDWEIGHT >= w.PPG) & (w.MUDWEIGHT <= w.FG_ml))[mm].mean() * 100)
        r['mw_below_pp'] = float((w.MUDWEIGHT < w.PPG)[mm].mean() * 100)
        r['mw_above_fg'] = float((w.MUDWEIGHT > w.FG_ml)[mm].mean() * 100)
    else:
        r['mw_between'] = r['mw_below_pp'] = r['mw_above_fg'] = np.nan
    return w, r


rows, store = [], {}
for wl in wells:
    wd, r = run_well(wl)
    rows.append(r); store[wl] = wd
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(W + 'geomech_perwell.csv', index=False)
good = tab[tab.trend_ok & (tab.n_mw > 500) & (tab.dts_n > 500)]  # measured shear needed so the whole chain (incl. nu) can be checked; criterion added after seeing the per-well table
summ = dict(n_eligible=int(len(tab)), n_trend_ok=int(tab.trend_ok.sum()),
            median_rho_r2=float(tab.rho_r2.median()), median_rho_mae=float(tab.rho_mae.median()),
            median_dts_r2=float(tab.dts_r2.median()), median_nu_mae=float(tab.nu_mae.median()),
            median_obg_diff_sg=float(tab.obg_diff_sg.median()), max_obg_diff_sg=float(tab.obg_diff_sg.max()))
# representative well for the figure: rule fixed in advance = identifiable trend and most mud-weight samples
rep = None
if len(good):
    rep = good.sort_values('n_mw', ascending=False).iloc[0].well
    summ['rep_well'] = rep
    summ['rep'] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in good[good.well == rep].iloc[0].to_dict().items()}
print(json.dumps(summ, indent=1, default=str))
json.dump(summ, open(W + 'geomech_summary.json', 'w'), indent=1, default=str)

if rep is not None:
    w = store[rep]; z = w.DEPTH_MD.values
    LC = ['#f2d16b', '#d9c58a', '#8d8d89', '#9bc4a8', '#b78fd1', '#7fb6e6', '#b9e0ef', '#f4a6c0', '#e76f51', '#444444', '#222222', '#c07a3d']
    fig, ax = plt.subplots(1, 8, figsize=(11, 8.5), sharey=True, gridspec_kw={'width_ratios': [0.3, 0.3, 1, 1, 1, 1, 1, 1.25]})
    for k, col, t in ((0, 'y', 'Expert\nlabel'), (1, 'rf', 'RF (CV)\nprediction')):
        vals = w[col].values
        for c in np.unique(vals[~np.isnan(vals)]):
            ax[k].fill_betweenx(z, 0, 1, where=(vals == c), color=LC[int(c)], step='mid', lw=0)
        ax[k].set_xticks([]); ax[k].set_title(t, fontsize=7)
    ax[2].plot(w.GR, z, color=GREEN, lw=0.5); ax[2].set_xlim(0, 200); ax[2].set_title('GR (API)', fontsize=8)
    ax[3].plot(w.DTC, z, color=BLUE, lw=0.5, label='DTC'); ax[3].plot(w.DTC_n, z, color=INK, lw=0.9, ls='--', label='normal trend')
    ax[3].set_xlim(40, 180); ax[3].set_title('Compressional\nslowness (µs/ft)', fontsize=8); ax[3].legend(fontsize=6, frameon=False, loc='lower left')
    ax[4].plot(w.RHOB, z, color=INK, lw=0.6, label='measured'); ax[4].plot(w.RHOB_ml, z, color=ORANGE, lw=0.5, label='ML (well held out)')
    ax[4].set_xlim(1.6, 3.0); ax[4].set_title('Bulk density\n(g/cm³)', fontsize=8); ax[4].legend(fontsize=6, frameon=False, loc='lower left')
    ax[5].plot(w.DTS, z, color=INK, lw=0.6, label='measured'); ax[5].plot(w.DTS_ml, z, color=ORANGE, lw=0.5, label='ML (well held out)')
    ax[5].set_title('Shear slowness\n(µs/ft)', fontsize=8); ax[5].legend(fontsize=6, frameon=False, loc='lower left')
    ax[6].plot(w.nu_meas, z, color=INK, lw=0.6, label='from measured DTS'); ax[6].plot(w.nu_ml, z, color=ORANGE, lw=0.5, label='from ML DTS')
    ax[6].set_xlim(0, 0.5); ax[6].set_title("Poisson's ratio", fontsize=8); ax[6].legend(fontsize=6, frameon=False, loc='lower left')
    ax[7].plot(w.OBG, z, color=GREY, lw=0.8, label='Overburden'); ax[7].plot(w.PPG, z, color=AQUA, lw=0.7, label='Eaton PPG (shale)')
    ax[7].plot(w.FG_ml, z, color=MAGENTA, lw=0.6, label='Eaton FG (ML ν)'); ax[7].plot(w.MUDWEIGHT, z, color=INK, lw=0.9, label='Mud weight used')
    ax[7].set_xlim(0.8, 2.6); ax[7].set_title('Gradients (SG)', fontsize=8); ax[7].legend(fontsize=6, frameon=False, loc='lower left')
    ax[0].set_ylabel('Measured depth (m)'); ax[0].set_ylim(z.max(), z.min())
    for a_ in ax:
        a_.grid(axis='x', lw=0.3, color='#dddddd')
        for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
    plt.tight_layout(); plt.savefig(W + 'figs/fig_open_well_geomech.png', dpi=300); plt.close()
    w.to_csv(W + 'geomech_rep_well.csv', index=False)
print('done')
