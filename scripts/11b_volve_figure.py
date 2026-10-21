import sys, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, volve_io as v
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import RESULTS
OUT = os.path.join(RESULTS, 'volve') + os.sep
FIG = os.path.join(RESULTS, 'figs') + os.sep
BLUE, ORANGE, AQUA, YELLOW, INK, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#222222', '#8a8a86'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555555', 'font.family': 'DejaVu Sans'})
fm, prof, pr, gas = pd.read_pickle(OUT + 'volve_objs.pkl')
col = {'F-1': BLUE, 'F-1 A': ORANGE, 'F-1 B': AQUA, 'F-1 C': YELLOW}
mk = {'F-1': 'o', 'F-1 A': 's', 'F-1 B': '^', 'F-1 C': 'D'}
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.4))
a = ax[0, 0]
for w, g in fm.groupby('well'):
    a.scatter(g.emw[~g.tight], g.tvd[~g.tight], s=22, color=col[w], marker=mk[w], label=f'{w} (n={len(g)})', zorder=3)
    a.scatter(g.emw[g.tight], g.tvd[g.tight], s=22, facecolors='none', edgecolors=col[w], marker=mk[w], zorder=3)
for w, g in fm.dropna(subset=['mw']).groupby('well'):
    a.plot([g.mw.iloc[0], g.mw.iloc[-1]] if g.mw.nunique() > 1 else [g.mw.iloc[0]] * 2, [g.tvd.min(), g.tvd.max()], color=col[w], lw=1.0, ls='--')
a.axvline(1.03, color=GREY, lw=0.8, ls=':'); a.text(1.032, 2935, 'seawater 1.03', fontsize=6.5, color=GREY, va='top')
a.invert_yaxis(); a.set_xlim(1.0, 1.42); a.set_xlabel('Equivalent mud weight (sg)'); a.set_ylabel('True vertical depth (m)')
a.set_title('(a) Measured pressure and mud weight', fontsize=8.5); a.legend(frameon=False, fontsize=6.3, loc='lower left', bbox_to_anchor=(0.0, 0.0))
b = ax[0, 1]
for w, mk_ in [('F-1', 'o'), ('F-1 B', '^'), ('F-1 C', 'D')]:
    m = v.mudlog(w); d = ((np.log10(m.ROPA * 3.28084 / (60 * m.RPMA)) / np.log10(12 * m.WOBA * 2204.62 / (1e6 * m.BDIA))) * 1.03 / m.MDIA)
    ok = (m.ROPA > 1) & (m.WOBA > 1) & (m.RPMA > 40) & m.DXC.between(0.3, 2) & d.between(0.3, 2)
    b.scatter(m.DXC[ok], d[ok], s=3, alpha=0.25, color=col[w], marker=mk_, label=w, rasterized=True)
b.plot([0.3, 2], [0.3, 2], color=INK, lw=0.7); b.set_xlim(0.3, 1.6); b.set_ylim(0.3, 1.6); b.set_xlabel('dc recorded in mud log (DXC)'); b.set_ylabel('dc recomputed here')
b.set_title('(b) d-exponent check', fontsize=8.5); b.legend(frameon=False, fontsize=6.5, markerscale=3, loc='upper left')
c = ax[1, 0]
pr2 = pr.copy()
c.scatter(pr2.emw, pr2['pp_2.0'], s=22, color=[col[w] for w in pr2.well], marker='o', zorder=3)
c.plot([1.0, 1.25], [1.0, 1.25], color=INK, lw=0.7); c.axhline(1.03, color=GREY, lw=0.8, ls=':')
c.set_xlim(1.0, 1.25); c.set_ylim(1.0, 1.25); c.set_xlabel('Measured pressure (sg)'); c.set_ylabel('Eaton d-exponent estimate (sg)')
c.text(1.005, 1.035, 'clamped at normal pressure 1.03', fontsize=6.5, color=GREY)
c.set_title('(c) Eaton estimate vs. measured', fontsize=8.5)
d_ = ax[1, 1]
data = []; labs = []
for w in ['F-1', 'F-1 B', 'F-1 C']:
    z = gas[w]; z = z[z.interp & (z.wh <= 40)].bern.dropna(); data.append(z.values); labs.append(f'{w}\n(n={len(z)})')
bp = d_.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
for p_, w in zip(bp['boxes'], ['F-1', 'F-1 B', 'F-1 C']): p_.set(facecolor=col[w], alpha=0.6, lw=0.6)
d_.set_xticklabels(labs, fontsize=7); d_.set_yscale('log'); d_.set_ylim(1, 150); d_.axhline(100, color=INK, lw=0.6, ls=':')
d_.text(0.55, 108, 'thermogenic below ~100', fontsize=6.5); d_.set_ylabel('C1 / (C2 + C3)'); d_.set_title('(d) Bernard parameter, 8.5 in section', fontsize=8.5)
for a_ in ax.ravel():
    for sp in ('top', 'right'): a_.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(FIG + 'fig_volve.png', dpi=300); plt.close()
