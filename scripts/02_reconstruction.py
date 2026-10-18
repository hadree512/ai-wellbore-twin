import os
"""Experiment B: reconstruct bulk density (RHOB) and shear slowness (DTS) from other logs.
Validation is leave-wells-out (GroupKFold), so no well appears in both train and test."""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

from paths import RESULTS as W, DATA
df = pd.read_csv(DATA + 'train.csv', sep=';',
                 usecols=['WELL', 'DEPTH_MD', 'GR', 'RHOB', 'NPHI', 'DTC', 'DTS', 'RDEP', 'CALI', 'DRHO'])
df['LRDEP'] = np.log10(df.RDEP.clip(lower=0.01))
# a little quality control: drop washed-out density readings and physically absurd values
df = df[(df.DRHO.abs() < 0.15) | df.DRHO.isna()]
df = df[df.RHOB.between(1.5, 3.2) | df.RHOB.isna()]
df = df[df.DTC.between(40, 200) | df.DTC.isna()]
df = df[df.DTS.between(70, 500) | df.DTS.isna()]


def run(target, feats, baseline, label, sub_n=150000, seed=0):
    d = df.dropna(subset=[target] + feats).copy()
    if len(d) > 600000:
        d = d.sample(600000, random_state=seed)
    X, y, g = d[feats].values, d[target].values, d.WELL.values
    print(label, 'rows', len(d), 'wells', len(np.unique(g)), flush=True)
    gkf = GroupKFold(n_splits=5)
    preds = {k: np.zeros(len(y)) for k in ['Baseline', 'MLP', 'XGBoost']}
    rng = np.random.RandomState(seed)
    for tr, te in gkf.split(X, y, g):
        preds['Baseline'][te] = baseline(d.iloc[tr], d.iloc[te])
        s = rng.choice(tr, min(sub_n, len(tr)), replace=False)
        mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=seed))
        mlp.fit(X[s], y[s]); preds['MLP'][te] = mlp.predict(X[te])
        gb = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.7, tree_method='hist', n_jobs=2, random_state=seed)
        gb.fit(X[s], y[s]); preds['XGBoost'][te] = gb.predict(X[te])
    out = {}
    for k, p in preds.items():
        pw = [r2_score(y[g == w], p[g == w]) for w in np.unique(g) if (g == w).sum() > 200]
        out[k] = dict(r2=r2_score(y, p), mae=mean_absolute_error(y, p), rmse=mean_squared_error(y, p) ** 0.5,
                      median_per_well_r2=float(np.median(pw)))
        print(' ', k, {a: round(b, 3) for a, b in out[k].items()}, flush=True)
    np.savez(W + f'recon_{target}.npz', y=y, g=g, depth=d.DEPTH_MD.values, **preds)
    return out, len(d), int(len(np.unique(g)))


def gardner(train, test):
    # rho = 0.23 * Vp^0.25, Vp in ft/s, Vp = 1e6 / DTC. Constants kept as published.
    return 0.23 * (1e6 / test.DTC.values) ** 0.25


def castagna_like(train, test):
    # DTS = a * DTC + b fitted on the training wells only
    lr = LinearRegression().fit(train[['DTC']].values, train.DTS.values)
    return lr.predict(test[['DTC']].values)


res = {}
res['RHOB'] = run('RHOB', ['GR', 'NPHI', 'DTC', 'LRDEP', 'DEPTH_MD', 'CALI'], gardner, 'density')
res['DTS'] = run('DTS', ['GR', 'RHOB', 'NPHI', 'DTC', 'LRDEP', 'DEPTH_MD'], castagna_like, 'shear slowness')

# Poisson's ratio derived from the measured vs predicted DTS
z = np.load(W + 'recon_DTS.npz')
d = df.dropna(subset=['DTS', 'RHOB', 'NPHI', 'DTC', 'GR', 'LRDEP']).copy()
if len(d) > 600000:
    d = d.sample(600000, random_state=0)
dtc = d.DTC.values


def nu(dts, dtc):
    r2 = (dts / dtc) ** 2
    return (0.5 * r2 - 1) / (r2 - 1)

nu_true = nu(z['y'], dtc)
pr = {}
for k in ['Baseline', 'MLP', 'XGBoost']:
    n_hat = nu(z[k], dtc)
    ok = np.isfinite(nu_true) & np.isfinite(n_hat) & (nu_true > 0) & (nu_true < 0.5)
    pr[k] = dict(r2=r2_score(nu_true[ok], n_hat[ok]), mae=mean_absolute_error(nu_true[ok], n_hat[ok]))
    print("Poisson's ratio", k, {a: round(b, 4) for a, b in pr[k].items()})
res['nu'] = pr
json.dump(res, open(W + 'recon_metrics.json', 'w'), indent=1, default=float)
np.savez(W + 'recon_nu.npz', nu_true=nu_true, **{k: nu(z[k], dtc) for k in ['Baseline', 'MLP', 'XGBoost']})
