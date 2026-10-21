"""Volve (Equinor open data): measured formation pressure vs mud weight and a d-exponent/Eaton estimate, plus mud-gas diagnostics.
Wells: 15/9-F-1 and sidetracks A, B, C. All inputs are read from the local Volve copy (see volve_io.py)."""
import sys, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, volve_io as v
from scipy.stats import theilslopes
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import RESULTS
OUT = os.path.join(RESULTS, 'volve') + os.sep; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, INK, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555', 'font.family': 'DejaVu Sans'})
PN, G = 1.03, 0.0980665
WELLS = ['F-1', 'F-1 A', 'F-1 B', 'F-1 C']
res = {}

# ---------- 1. measured pressures, mud weight overbalance, cross-well consistency ----------
rows = []
for w in WELLS:
    s, f = v.survey(w), v.fm(w)
    f['well'] = w; f['tvd'] = np.interp(f.md, s.md, s.tvd); f['emw'] = f.p_bar / (G * f.tvd)
    if w in v.MUD:
        m = v.mudlog(w); f['mw'] = np.interp(f.md, m.TDEP, m.MDIA.interpolate())
    else: f['mw'] = np.nan
    rows.append(f)
fm = pd.concat(rows, ignore_index=True); fm['over_sg'] = fm.mw - fm.emw
fm['tight'] = fm.mob < 10
fm.to_csv(OUT + 'volve_fm_tests.csv', index=False)
def fit(d):
    b, a = np.polyfit(d.tvd, d.p_bar, 1); r = d.p_bar - (a + b * d.tvd)
    return dict(n=int(len(d)), grad_sg=float(b / G), rms_bar=float(np.sqrt((r ** 2).mean())), emw_min=float(d.emw.min()), emw_max=float(d.emw.max()), emw_med=float(d.emw.median()))
res['fm_all'] = fit(fm); res['fm_mobile'] = fit(fm[~fm.tight])
res['fm_per_well'] = {w: dict(n=int(len(g)), emw_min=float(g.emw.min()), emw_max=float(g.emw.max()), tvd_min=float(g.tvd.min()), tvd_max=float(g.tvd.max()),
                              n_tight=int(g.tight.sum())) for w, g in fm.groupby('well')}
grads = {}
for w, g in fm[~fm.tight].groupby('well'):
    if len(g) >= 3:
        b, a = np.polyfit(g.tvd, g.p_bar, 1); r = g.p_bar - (a + b * g.tvd)
        grads[w] = dict(n=int(len(g)), grad_sg=float(b / G), r2=float(1 - (r ** 2).sum() / ((g.p_bar - g.p_bar.mean()) ** 2).sum()), span_m=float(g.tvd.max() - g.tvd.min()))
res['fm_grad'] = grads
o = fm.dropna(subset=['over_sg'])
res['overbalance'] = dict(n=int(len(o)), min=float(o.over_sg.min()), med=float(o.over_sg.median()), max=float(o.over_sg.max()),
                          mw_min=float(o.mw.min()), mw_max=float(o.mw.max()))

# ---------- 2. d-exponent from mud logs, normal trend, Eaton ----------
def dcalc(m):
    R = m.ROPA * 3.28084; N = m.RPMA; Wlb = m.WOBA * 2204.62
    d = np.log10(R / (60 * N)) / np.log10(12 * Wlb / (1e6 * m.BDIA))
    return d.where((m.ROPA > 1) & (m.WOBA > 1) & (m.RPMA > 40) & np.isfinite(d))

def obg(tvd, rho, rho_top):
    rho = pd.Series(rho).interpolate(limit_direction='both').values
    dz = np.diff(tvd, prepend=tvd[0]); return (rho_top * tvd[0] + np.cumsum(rho * dz)) / tvd

dcres, pred_rows = {}, []
prof = {}
for w in ['F-1', 'F-1 B', 'F-1 C']:
    m = v.mudlog(w).dropna(subset=['DVER']); c = v.composite(w)
    m['d'] = dcalc(m); m['dc'] = m.d * PN / m.MDIA
    ok = m.d.notna() & m.DXC.notna() & (m.DXC > 0)
    dcres[w] = dict(n=int(ok.sum()), corr=float(np.corrcoef(m.dc[ok], m.DXC[ok])[0, 1]), ratio_med=float((m.dc[ok] / m.DXC[ok]).median()))
    m['gr'] = np.interp(m.TDEP, c.md[c.GR.notna()], c.GR.dropna())
    m['shale'] = m.gr > 75
    # robust 25 m median of shale dc
    sh = m[m.shale & m.dc.notna() & m.dc.between(0.3, 2.0)].copy()
    sh['bin'] = (sh.TDEP // 25) * 25 + 12.5
    p = sh.groupby('bin').agg(tvd=('DVER', 'median'), dc=('dc', 'median'), n=('dc', 'size'), mw=('MDIA', 'median')).query('n>=5').reset_index()
    # normal trend: ln(dc) vs TVD in the 17.5 in section (1300-2400 m MD), where a compaction response is expected
    nt = p[(p.bin >= 1300) & (p.bin <= 2400)]
    if len(nt) < 8: continue
    b, a, _, _ = theilslopes(np.log(nt.dc), nt.tvd)
    p['dcn'] = np.exp(a + b * p.tvd)
    # overburden gradient (sg) from composite density in the logged interval, assumed rho_top above it
    cc = c[c.DEN.between(1.5, 3.2)]
    for rt in (1.9, 2.0, 2.2):
        full = np.arange(c.tvd.min(), c.tvd.max(), 1.0)
        rho = np.interp(full, cc.tvd, cc.DEN, left=np.nan, right=np.nan)
        # below first logged density use the log; above it use the assumed constant
        first = cc.tvd.min(); rho = np.where(full < first, rt, rho)
        S = np.interp(p.tvd, full, obg(full, rho, rt))
        p[f'pp_{rt}'] = np.maximum(S - (S - PN) * (p.dc / p.dcn) ** 1.2, PN)
    p['slope_b'] = b
    prof[w] = p; dcres[w].update(trend_slope=float(b), n_trend_pts=int(len(nt)))
    # compare with measured pressure at the test depths (nearest shale bins within 60 m TVD)
    g = fm[fm.well == w]
    for _, r in g.iterrows():
        near = p[(p.tvd - r.tvd).abs() <= 60]
        if len(near) >= 1:
            pred_rows.append(dict(well=w, tvd=r.tvd, emw=r.emw, tight=r.tight, n_bins=len(near), **{k: near[k].median() for k in ['pp_1.9', 'pp_2.0', 'pp_2.2']}, dcdcn=float((near.dc / near.dcn).median())))
pr = pd.DataFrame(pred_rows); pr.to_csv(OUT + 'volve_eaton_dc_vs_measured.csv', index=False)
res['dc_check'] = dcres
if len(pr):
    for k in ['pp_1.9', 'pp_2.0', 'pp_2.2']:
        e = pr[k] - pr.emw
        res[f'eaton_dc_{k}'] = dict(n=int(len(pr)), bias=float(e.mean()), mae=float(e.abs().mean()), rmse=float(np.sqrt((e ** 2).mean())), pred_med=float(pr[k].median()))
    res['eaton_meas_med'] = float(pr.emw.median())
    res['eaton_dcdcn_med'] = float(pr.dcdcn.median())

# ---------- 3. gas diagnostics in the 8.5 in section (three mud logs) ----------
gas_rows = []; gas_ratio = {}
for w in ['F-1', 'F-1 B', 'F-1 C']:
    m = v.mudlog(w)
    z = m[(m.BDIA <= 8.6) & (m.TDEP > 2400)].copy()
    c1, c2, c3 = z.MTHA, z.ETHA, z.PRPA
    ic4, nc4, ic5, nc5 = z.IBTA, z.NBTA, z.IPNA, z.NPNA
    c4, c5 = ic4 + nc4, ic5 + nc5
    z['bern'] = c1 / (c2 + c3)
    z['wh'] = (c2 + c3 + c4 + c5) / (c1 + c2 + c3 + c4 + c5) * 100
    z['bh'] = (c1 + c2) / (c3 + c4 + c5)
    z['ch'] = (c4 + c5) / c3
    r2, r3, r4, r5 = c1 / c2, c1 / c3, c1 / c4, c1 / c5
    z['pix345'] = (r2 < r3) & (r3 < r4)
    z['ic4nc4'] = ic4 / nc4
    z['tg'] = z.GASA
    ref = float(z.ROPA.median()); z['tg_norm'] = z.tg * ref / z.ROPA.clip(lower=1)
    z['sum15'] = c1 + c2 + c3 + c4 + c5
    z['interp'] = (z.sum15 > 3000) & (c2 > 0) & (c3 > 0)
    zz = z[z.interp].dropna(subset=['bern', 'wh'])
    zz = zz[zz.wh <= 40]
    gas_rows.append(dict(well=w, n=int(len(zz)), n_8p5=int(len(z)), pct_interp=float(100 * z.interp.mean()), tg_med=float(z.tg.median()), tg_max=float(z.tg.max()), c1_med=float(c1.median()), bern_med=float(zz.bern.median()),
                         wh_min=float(zz.wh.min()), wh_max=float(zz.wh.max()), bern_min=float(zz.bern.min()), bern_max=float(zz.bern.max()), bh_gt_wh=float(100 * (zz.bh > zz.wh).mean()), wh_med=float(zz.wh.median()), bh_med=float(zz.bh.median()), ch_med=float(zz.ch.median()), pix345=float(100 * zz.pix345.mean()),
                         ic4nc4_med=float(zz.ic4nc4.median()), rop_ref=ref,
                         corr_tg_rop=float(z[['tg', 'ROPA']].corr(method='spearman').iloc[0, 1]), corr_tgn_rop=float(z[['tg_norm', 'ROPA']].corr(method='spearman').iloc[0, 1])))
    gas_ratio[w] = z
gt = pd.DataFrame(gas_rows); gt.to_csv(OUT + 'volve_gas_diag.csv', index=False)
res['gas'] = gas_rows
json.dump(res, open(OUT + 'volve_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1)[:6000])
print(fm.round(3).to_string()); print(pr.round(3).to_string())
pd.to_pickle((fm, prof, pr, gas_ratio), OUT + 'volve_objs.pkl')   # used by 11b_volve_figure.py
