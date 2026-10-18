import os
"""Experiment A: lithology classification on FORCE 2020 open wells, leave-wells-out CV."""
import json, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import xgboost as xgb

from paths import RESULTS as W, DATA
df = pd.read_csv(DATA + 'train.csv', sep=';')
codes = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}
names = ['Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']
df['y'] = df.FORCE_2020_LITHOFACIES_LITHOLOGY.map(codes)
A = np.load(DATA + 'penalty_matrix.npy')

df['LRDEP'] = np.log10(df.RDEP.clip(lower=0.01))
df['LRMED'] = np.log10(df.RMED.clip(lower=0.01))
feats = ['DEPTH_MD', 'GR', 'RHOB', 'NPHI', 'DTC', 'LRDEP', 'LRMED', 'CALI', 'DRHO', 'PEF']
# keep rows with at least GR (always present); others may be NaN
X = df[feats].values.astype(float)
y = df.y.values
g = df.WELL.values

def score(yt, yp):
    return -A[yt, yp].mean()

def empirical(d):
    """Conventional threshold rules on GR / RHOB / NPHI / DTC (textbook cut-offs)."""
    gr, rho, nphi, dtc = d.GR.values, d.RHOB.values, d.NPHI.values, d.DTC.values
    out = np.full(len(d), 2)                      # default: shale
    sand = gr < 60
    out[sand] = 0                                 # sandstone
    mid = (gr >= 60) & (gr < 90)
    out[mid] = 1                                  # interbedded sand/shale
    carb = (gr < 45) & (rho > 2.55)
    out[carb & (rho < 2.80)] = 5                  # limestone
    out[(gr < 45) & (rho >= 2.80) & (rho < 2.95)] = 4   # dolomite
    out[(gr < 30) & (np.abs(rho - 2.03) < 0.08) & (dtc < 75)] = 7   # halite
    out[(rho >= 2.9) & (gr < 30)] = 8             # anhydrite
    return out

wells = np.unique(g)
gkf = GroupKFold(n_splits=5)
rng = np.random.RandomState(0)

models = {
    'Empirical thresholds': None,
    'Random Forest': lambda: make_pipeline(SimpleImputer(strategy='median'),
                                           RandomForestClassifier(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=0, class_weight='balanced_subsample', max_samples=0.3)),
    'XGBoost': lambda: xgb.XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.15, subsample=0.5, tree_method='hist', n_jobs=-1, random_state=0),
    'SVM (RBF)': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), SVC(C=5, gamma='scale', class_weight='balanced')),
}
res = {k: {'pred': np.full(len(y), -1)} for k in models}
t0 = time.time()
for fold, (tr, te) in enumerate(gkf.split(X, y, g)):
    print('fold', fold, len(tr), len(te), 'test wells', len(np.unique(g[te])), flush=True)
    # present-class remap for xgboost
    cls = np.unique(y[tr]); remap = {c: i for i, c in enumerate(cls)}; inv = cls
    for name, mk in models.items():
        if mk is None:
            res[name]['pred'][te] = empirical(df.iloc[te]); continue
        if name == 'SVM (RBF)':
            sub = rng.choice(tr, 30000, replace=False)       # SVM scales poorly -> subsample
            m = mk(); m.fit(X[sub], y[sub]); p = m.predict(X[te])
        elif name == 'XGBoost':
            sub = rng.choice(tr, 400000, replace=False)
            m = mk(); m.fit(X[sub], np.vectorize(remap.get)(y[sub])); p = inv[m.predict(X[te])]
        else:
            sub = rng.choice(tr, 400000, replace=False)
            m = mk(); m.fit(X[sub], y[sub]); p = m.predict(X[te])
        res[name]['pred'][te] = p
    print('  elapsed', round(time.time() - t0), flush=True)

# metrics: pooled and per-well
out = {}
for name in models:
    p = res[name]['pred']
    per_well_acc = [accuracy_score(y[g == w], p[g == w]) for w in wells]
    out[name] = dict(accuracy=accuracy_score(y, p), balanced_acc=balanced_accuracy_score(y, p),
                     macro_f1=f1_score(y, p, average='macro'), force_score=score(y, p),
                     per_well_acc_mean=float(np.mean(per_well_acc)), per_well_acc_sd=float(np.std(per_well_acc)))
    print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
json.dump(out, open(W + 'litho_metrics.json', 'w'), indent=1)
np.save(W + 'litho_pred_rf.npy', res['Random Forest']['pred'])
np.save(W + 'litho_pred_xgb.npy', res['XGBoost']['pred'])
np.save(W + 'litho_pred_emp.npy', res['Empirical thresholds']['pred'])
np.save(W + 'litho_pred_svm.npy', res['SVM (RBF)']['pred'])
# per-class F1 for best model
best = max(out, key=lambda k: out[k]['macro_f1'])
f1c = f1_score(y, res[best]['pred'], average=None, labels=range(12))
json.dump({'best': best, 'per_class_f1': dict(zip(names, map(float, f1c))),
           'class_counts': {names[i]: int((y == i).sum()) for i in range(12)}}, open(W + 'litho_perclass.json', 'w'), indent=1)
# feature importance from RF refit on a subsample
m = models['Random Forest'](); sub = rng.choice(len(y), 400000, replace=False); m.fit(X[sub], y[sub])
json.dump(dict(zip(feats, map(float, m[-1].feature_importances_))), open(W + 'litho_featimp.json', 'w'), indent=1)
np.save(W + 'litho_cm_best.npy', confusion_matrix(y, res[best]['pred'], labels=range(12)))
print('done', round(time.time() - t0))
