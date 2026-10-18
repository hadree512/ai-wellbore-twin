"""Extended geomechanics for the eight eligible open wells: dynamic elastic moduli, empirical strength,
horizontal stress, frictional-limit check and a vertical-well stability window.
Input: geomech_all_wells.csv (written by the pore-pressure/fracture-gradient script).
Everything here is an estimate from empirical relations; nothing is calibrated to core or leak-off data."""
import json
import numpy as np
import pandas as pd

from paths import RESULTS as W, DATA, PRIVATE
d = pd.read_csv(W + 'geomech_all_wells.csv')
G_ACC = 0.00981            # MPa per metre per SG
MU = 0.6                   # sliding friction coefficient for the frictional-limit check (Byerlee-type value)
Q_F = (np.sqrt(MU ** 2 + 1) + MU) ** 2        # about 3.12

# ---- inputs: measured shear slowness where present, ML shear slowness elsewhere
d['rho'] = d.RHOB.where(d.RHOB.between(1.5, 3.2)).fillna(d.RHOB_ml)
d['dts'] = d.DTS.where(d.DTS.between(70, 500)).fillna(d.DTS_ml)
d['shear_src'] = np.where(d.DTS.between(70, 500), 'measured', 'ML')
ok = d.rho.notna() & d.dts.notna() & d.DTC.between(40, 200)
d = d[ok].copy()
d['vp'] = 304800.0 / d.DTC           # m/s  (slowness in us/ft)
d['vs'] = 304800.0 / d.dts
d = d[d.vp > d.vs * np.sqrt(4 / 3)]  # physically admissible: Vp^2 > 4/3 Vs^2
rho = d.rho * 1000.0                 # kg/m3
d['G'] = rho * d.vs ** 2 / 1e9       # GPa
d['K'] = rho * (d.vp ** 2 - 4 / 3 * d.vs ** 2) / 1e9
d['E'] = 9 * d.K * d.G / (3 * d.K + d.G)
d['nu'] = (3 * d.K - 2 * d.G) / (2 * (3 * d.K + d.G))

# ---- empirical strength (shale only; GR > 75 API)
vp_kms = d.vp / 1000.0
shale = d.GR > 75
d['UCS'] = np.where(shale, 0.77 * vp_kms ** 2.93, np.nan)                  # MPa, Horsrud (2001) shale relation
d['phi'] = np.where(shale, np.degrees(np.arcsin((vp_kms - 1) / (vp_kms + 1))), np.nan)  # Lal (1999) shale friction angle

# ---- stresses, in SG-equivalent gradients (measured depth taken as vertical depth)
D = d.DEPTH_MD
sv, pp = d.OBG, d.PPG
nu_use = d.nu_meas.fillna(d.nu_ml)
d['shmin'] = pp + nu_use / (1 - nu_use) * (sv - pp)          # poroelastic, no tectonic strain, Biot coefficient 1
d['shmin_lim'] = pp + (sv - pp) / Q_F                         # lowest Shmin that frictional faulting allows
d['crit'] = d.shmin < d.shmin_lim
d['eff_sv'] = sv - pp                                         # Terzaghi effective vertical stress, SG

# ---- vertical-well stability window, isotropic horizontal stress (SHmax = Shmin), tensile strength zero
ucs_sg = d.UCS / (G_ACC * D)
q = np.tan(np.radians(45 + d.phi / 2)) ** 2
d['collapse'] = (2 * d.shmin - ucs_sg - (1 - q) * pp) / (1 + q)
d['frac_init'] = 2 * d.shmin - pp
d['collapse'] = d.collapse.clip(lower=pp)                     # a collapse limit below pore pressure is not meaningful

rows = []
for w, g in d.groupby('WELL'):
    s = g[g.PPG.notna()]
    r = dict(well=w, n=int(len(g)), n_meas_shear=int((g.shear_src == 'measured').sum()),
             E_med=float(g.E.median()), G_med=float(g.G.median()), K_med=float(g.K.median()), nu_med=float(g.nu.median()),
             ucs_med=float(g.UCS.median()), phi_med=float(g.phi.median()),
             n_stress=int(len(s)))
    if len(s) > 200:
        mw = s[s.MUDWEIGHT.between(0.9, 2.6)]
        r.update(shmin_med=float(s.shmin.median()), crit_pct=float(100 * s.crit.mean()),
                 collapse_med=float(s.collapse.median()) if s.collapse.notna().any() else np.nan,
                 frac_med=float(s.frac_init.median()))
        m2 = mw[mw.collapse.notna()]
        if len(m2) > 50:
            r.update(n_mw=int(len(m2)), mw_above_collapse=float(100 * (m2.MUDWEIGHT >= m2.collapse).mean()),
                     mw_below_frac=float(100 * (m2.MUDWEIGHT <= m2.frac_init).mean()),
                     mw_in_window=float(100 * ((m2.MUDWEIGHT >= m2.collapse) & (m2.MUDWEIGHT <= m2.frac_init)).mean()))
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(W + 'geomech_ext_perwell.csv', index=False)
print(tab.round(2).to_string())
summ = dict(frictional_ratio=float(Q_F), mu=MU, n_samples=int(len(d)), n_wells=int(d.WELL.nunique()),
            overall_crit_pct=float(100 * d[d.PPG.notna()].crit.mean()),
            E_p5_p95=[float(d.E.quantile(.05)), float(d.E.quantile(.95))], nu_p5_p95=[float(d.nu.quantile(.05)), float(d.nu.quantile(.95))])
json.dump(summ, open(W + 'geomech_ext_summary.json', 'w'), indent=1)
print(summ)
d.to_csv(W + 'geomech_ext_samples.csv', index=False)


# ---- caliper check (wells with bit size, mud weight and a collapse estimate): does the margin predict hole enlargement?
from scipy.stats import spearmanr
m = d.PPG.notna() & d.collapse.notna() & d.MUDWEIGHT.between(0.9, 2.6) & d.CALI.between(4, 40) & d.BS.between(4, 40)
s = d[m].copy(); s['enl'] = s.CALI - s.BS; s['margin'] = s.MUDWEIGHT - s.collapse
cal = {}
for w, g in s.groupby('WELL'):
    rr, pv = spearmanr(g.margin, g.enl)
    cal[w] = dict(n=int(len(g)), spearman=float(rr), p=float(pv), enl_below=float(g[g.margin < 0].enl.median()), enl_above=float(g[g.margin >= 0].enl.median()),
                  n_below=int((g.margin < 0).sum()), n_above=int((g.margin >= 0).sum()))
# the margin is tied to depth through the stepped mud weight, so also look inside intervals of constant mud weight
s = s.sort_values(['WELL', 'DEPTH_MD'])
s['mwstep'] = (s.MUDWEIGHT.diff().abs() > 1e-6).cumsum()
steps = [spearmanr(g.margin, g.enl)[0] for _, g in s.groupby('mwstep') if len(g) > 200]
for w in cal:
    gg = s[s.WELL == w]
    cal[w].update(depth_corr_margin=float(spearmanr(gg.DEPTH_MD, gg.margin)[0]), depth_corr_enl=float(spearmanr(gg.DEPTH_MD, gg.enl)[0]),
                  step_rho=[float(x) for x in steps], step_rho_median=float(np.median(steps)), n_steps=len(steps))
json.dump(cal, open(W + 'geomech_ext_caliper.json', 'w'), indent=1)
print(cal)

# ---- figure for the well with measured shear slowness
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, INK, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'font.family': 'DejaVu Sans'})
rep = json.load(open(W + 'geomech_summary.json'))['rep_well']
g = d[d.WELL == rep].sort_values('DEPTH_MD')
z = g.DEPTH_MD.values
fig, ax = plt.subplots(1, 7, figsize=(11, 8.4), sharey=True, gridspec_kw={'width_ratios': [1, 1.1, 1, 1, 1.2, 1.2, 1]})
ax[0].plot(g.GR, z, color=GREEN, lw=0.6); ax[0].set_xlim(0, 200); ax[0].axvline(75, color=GREY, lw=0.6, ls=':'); ax[0].set_title('GR (API)', fontsize=8)
for col, c, lab in [('E', BLUE, "Young's E"), ('G', ORANGE, 'Shear G'), ('K', AQUA, 'Bulk K')]:
    ax[1].plot(g[col], z, color=c, lw=0.6, label=lab)
ax[1].set_xlim(0, 60); ax[1].legend(frameon=False, fontsize=6.5, loc='lower right'); ax[1].set_title('Dynamic moduli\n(GPa)', fontsize=8)
ax[2].scatter(g.UCS, z, s=1.2, color=MAGENTA, lw=0); ax[2].set_xlim(0, 60); ax[2].set_title('UCS, shale\n(MPa)', fontsize=8)
ax[3].scatter(g.phi, z, s=1.2, color=YELLOW, lw=0); ax[3].set_xlim(10, 45); ax[3].set_title('Friction angle,\nshale (°)', fontsize=8)
ax[4].plot(g.OBG, z, color=GREY, lw=0.8, label='Overburden'); ax[4].plot(g.shmin, z, color=BLUE, lw=0.6, label='Shmin (Eaton)')
ax[4].plot(g.shmin_lim, z, color=ORANGE, lw=0.6, label='Frictional lower limit'); ax[4].plot(g.PPG, z, color=AQUA, lw=0.6, label='Pore pressure')
ax[4].set_xlim(0.9, 2.4); ax[4].legend(frameon=False, fontsize=6, loc='lower left'); ax[4].set_title('Stress gradients\n(SG)', fontsize=8)
ax[5].plot(g.collapse, z, color=ORANGE, lw=0.6, label='Collapse'); ax[5].plot(g.PPG, z, color=AQUA, lw=0.6, label='Pore pressure')
ax[5].plot(g.MUDWEIGHT, z, color=INK, lw=0.9, label='Mud weight used'); ax[5].set_xlim(0.9, 2.0); ax[5].legend(frameon=False, fontsize=6, loc='lower left')
ax[5].set_title('Stability window,\nvertical well (SG)', fontsize=8)
ax[6].plot((g.CALI - g.BS), z, color=INK, lw=0.5); ax[6].axvline(0, color=GREY, lw=0.6, ls=':'); ax[6].set_xlim(-1, 6); ax[6].set_title('Caliper minus\nbit size (in)', fontsize=8)
ax[0].set_ylim(z.max(), z.min()); ax[0].set_ylabel('Measured depth (m)')
for a in ax:
    a.grid(color='#dddddd', lw=0.4)
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(W + 'figs/fig_geomech_extended.png', dpi=300); plt.close()
print('fig ok')
