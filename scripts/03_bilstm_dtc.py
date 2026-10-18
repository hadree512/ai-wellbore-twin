import os
"""Experiment C: Bi-LSTM reconstruction of the compressional sonic log (DTC) from GR, RHOB, NPHI,
resistivity and depth. Test wells are never seen in training (3-fold leave-wells-out)."""
import json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
from paths import RESULTS as W, DATA
df = pd.read_csv(DATA + 'train.csv', sep=';', usecols=['WELL', 'DEPTH_MD', 'GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'DRHO'])
df['LRDEP'] = np.log10(df.RDEP.clip(lower=0.01))
df = df.dropna(subset=['GR', 'RHOB', 'NPHI', 'DTC', 'LRDEP'])
df = df[df.RHOB.between(1.5, 3.2) & df.DTC.between(40, 200) & (df.DRHO.abs().fillna(0) < 0.15)]
feats = ['GR', 'RHOB', 'NPHI', 'LRDEP', 'DEPTH_MD']
wells = df.WELL.unique()
print('wells with all five logs:', len(wells), 'rows', len(df), flush=True)

L = 200   # window length in samples (~30 m at 0.15 m spacing)

def windows(d, stride):
    Xs, ys, ws = [], [], []
    for w, grp in d.groupby('WELL'):
        a = grp.sort_values('DEPTH_MD')
        x, t = a[feats].values.astype(np.float32), a.DTC.values.astype(np.float32)
        for i in range(0, len(a) - L, stride):
            Xs.append(x[i:i + L]); ys.append(t[i:i + L]); ws.append(w)
    return np.array(Xs), np.array(ys), np.array(ws)

class BiLSTM(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.rnn = nn.LSTM(n_in, 48, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(96, 1)
    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)

gkf = GroupKFold(n_splits=3)
res = {'BiLSTM': [], 'XGBoost (point-wise)': [], 'Gardner-inverse baseline': []}
store = {}
t0 = time.time()
for fold, (tr_i, te_i) in enumerate(gkf.split(wells, groups=wells)):
    trw, tew = wells[tr_i], wells[te_i]
    dtr, dte = df[df.WELL.isin(trw)], df[df.WELL.isin(tew)]
    mu, sd = dtr[feats].mean().values.astype(np.float32), dtr[feats].std().values.astype(np.float32)
    tm, ts = float(dtr.DTC.mean()), float(dtr.DTC.std())
    Xtr, ytr, _ = windows(dtr, 100)
    Xte, yte, wte = windows(dte, 200)
    rs = np.random.RandomState(fold)
    if len(Xtr) > 5000:
        k = rs.choice(len(Xtr), 5000, replace=False); Xtr, ytr = Xtr[k], ytr[k]
    Xn = torch.tensor((Xtr - mu) / sd); yn = torch.tensor((ytr - tm) / ts)
    model = BiLSTM(len(feats)); opt = torch.optim.Adam(model.parameters(), 2e-3); lossf = nn.MSELoss()
    for ep in range(10):
        perm = torch.randperm(len(Xn)); tot = 0
        for b in range(0, len(Xn), 64):
            idx = perm[b:b + 64]
            opt.zero_grad(); l = lossf(model(Xn[idx]), yn[idx]); l.backward(); opt.step(); tot += l.item() * len(idx)
        print(f' fold {fold} epoch {ep} train mse {tot / len(Xn):.4f}', flush=True)
    model.eval()
    with torch.no_grad():
        p_lstm = model(torch.tensor((Xte - mu) / sd)).numpy() * ts + tm
    # point-wise XGBoost on the same test windows
    gb = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.7, tree_method='hist', n_jobs=2, random_state=0)
    s = dtr.sample(min(200000, len(dtr)), random_state=0)
    gb.fit(s[feats].values, s.DTC.values)
    p_xgb = gb.predict(Xte.reshape(-1, len(feats))).reshape(yte.shape)
    # Gardner inverted: rho = 0.23 * V^0.25  ->  V = (rho/0.23)^4, DTC = 1e6 / V
    p_gar = 1e6 / ((Xte[:, :, 1] / 0.23) ** 4)
    for name, p in [('BiLSTM', p_lstm), ('XGBoost (point-wise)', p_xgb), ('Gardner-inverse baseline', p_gar)]:
        res[name].append(dict(r2=r2_score(yte.ravel(), p.ravel()), mae=mean_absolute_error(yte.ravel(), p.ravel()),
                              rmse=mean_squared_error(yte.ravel(), p.ravel()) ** 0.5, n_test_wells=len(tew)))
        print(fold, name, {a: round(b, 3) for a, b in res[name][-1].items()}, flush=True)
    if fold == 0:
        store = dict(y=yte, lstm=p_lstm, xgb=p_xgb, gar=p_gar, X=Xte, w=wte)
    print(' elapsed', round(time.time() - t0), flush=True)

summary = {k: {m: float(np.mean([r[m] for r in v])) for m in ['r2', 'mae', 'rmse']} | {'r2_by_fold': [r['r2'] for r in v]} for k, v in res.items()}
json.dump(summary, open(W + 'bilstm_metrics.json', 'w'), indent=1)
np.savez(W + 'bilstm_fold0.npz', **store)
print(json.dumps(summary, indent=1))
