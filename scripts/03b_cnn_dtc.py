"""Experiment C2: 1-D CNN reconstruction of the compressional sonic log (DTC) from GR, RHOB, NPHI,
resistivity and depth. Test wells are never seen in training (3-fold leave-wells-out)."""
import json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
from paths import RESULTS as W, DATA, PRIVATE
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

class CNN1D(nn.Module):
    """Dilated 1-D convolutions; receptive field about 60 samples."""
    def __init__(self, n_in):
        super().__init__()
        layers, c = [], n_in
        for dil in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, 48, 5, padding=2 * dil, dilation=dil), nn.ReLU()]
            c = 48
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(48, 1, 1)
    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))).squeeze(1)

gkf = GroupKFold(n_splits=3)
res = []
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
    model = CNN1D(len(feats)); opt = torch.optim.Adam(model.parameters(), 2e-3); lossf = nn.MSELoss()
    for ep in range(10):
        perm = torch.randperm(len(Xn)); tot = 0
        for b in range(0, len(Xn), 64):
            idx = perm[b:b + 64]
            opt.zero_grad(); l = lossf(model(Xn[idx]), yn[idx]); l.backward(); opt.step(); tot += l.item() * len(idx)
        print(f' fold {fold} epoch {ep} train mse {tot / len(Xn):.4f}', flush=True)
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor((Xte - mu) / sd)).numpy() * ts + tm
    res.append(dict(r2=r2_score(yte.ravel(), p.ravel()), mae=mean_absolute_error(yte.ravel(), p.ravel()),
                    rmse=mean_squared_error(yte.ravel(), p.ravel()) ** 0.5))
    print(fold, {a: round(b, 3) for a, b in res[-1].items()}, 'elapsed', round(time.time() - t0), flush=True)
    if fold == 0:
        np.save(W + 'cnn_fold0_pred.npy', p)

out = {m: float(np.mean([r[m] for r in res])) for m in ['r2', 'mae', 'rmse']} | {'r2_by_fold': [r['r2'] for r in res]}
json.dump(out, open(W + 'cnn_metrics.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
