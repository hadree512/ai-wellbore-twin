"""Readers for a local copy of the Equinor Volve data (not redistributed here; see README)."""
"""Readers for the local Volve copy (staged under /mnt/user-data/uploads/datasets/volve)."""
import re, numpy as np, pandas as pd
from dlisio import dlis
import os
from paths import DATA
# Local copy of the Volve release (Equinor ASA, 2018). Set VOLVE to the folder that holds wells/<wellbore>/...
V = os.path.join(os.environ.get('VOLVE', os.path.join(DATA, 'volve')), 'wells') + os.sep
MUD = {'F-1': '15_9-F-1/mud_log/MUD_LOG_2.DLIS',
       'F-1 B': '15_9-F-1_B/mud_log/MUD_LOG_2.DLIS',
       'F-1 C': '15_9-F-1_C/mud_log/MUD_LOG_1.DLIS'}
SURV = {'F-1': '15_9-F-1/survey/Volve F_F-1_F-1_F-1_ACTUAL',
        'F-1 A': '15_9-F-1/survey/15_9-F-1 A/Volve F_F-1_F-1 A_F-1 A_ACTUAL',
        'F-1 B': '15_9-F-1/survey/15_9-F-1 B/Volve F_F-1_F-1 B_F-1 B_ACTUAL',
        'F-1 C': '15_9-F-1/survey/15_9-F-1 C/Volve F_F-1_F-1 C_F-1 C_ACTUAL'}
FM = {'F-1': '15_9-F-1/formation_pressure/FM_PRESS_COMPUTED_MWD_1.LAS',
      'F-1 A': '15_9-F-1_A/formation_pressure/FM_PRESS_COMPUTED_MWD_1.LAS',
      'F-1 B': '15_9-F-1_B/formation_pressure/FM_PRESS_COMPUTED_MWD_1.LAS',
      'F-1 C': '15_9-F-1_C/formation_pressure/FM_PRESS_COMPUTED_MWD_1.LAS'}

def mudlog(well):
    f, *_ = dlis.load(V + MUD[well])
    fr = f.frames[0]; c = fr.curves()
    d = pd.DataFrame({n: np.asarray(c[n]).reshape(len(c), -1)[:, 0].astype(float) for n in c.dtype.names})
    return d.replace(-999.25, np.nan)

def survey(well):
    rows = []
    for ln in open(V + SURV[well], encoding='latin-1'):
        t = ln.split()
        if len(t) >= 4:
            try: rows.append([float(x) for x in t[:4]])
            except ValueError: pass
    return pd.DataFrame(rows, columns=['md', 'inc', 'azi', 'tvd'])

def fm(well):
    rows = []; on = False
    for ln in open(V + FM[well]):
        if ln.startswith('~A'): on = True; hdr = ln.split()[1:]; continue
        if on and ln.strip(): rows.append([float(x) for x in ln.split()])
    d = pd.DataFrame(rows, columns=hdr).rename(columns={'DEPTH': 'md', 'PRES_FORM': 'p_bar', 'PRES_QUAL': 'q', 'PRES_MOBL': 'mob'})
    return d

COMP = {'F-1': '15_9-F-1', 'F-1 A': '15_9-F-1_A', 'F-1 B': '15_9-F-1_B', 'F-1 C': '15_9-F-1_C'}

def composite(well):
    f, *_ = dlis.load(V + COMP[well] + '/composite_logs/WLC_COMPOSITE_1.DLIS')
    c = f.frames[0].curves()
    d = pd.DataFrame({n: np.asarray(c[n]).reshape(len(c), -1)[:, 0].astype(float) for n in c.dtype.names if n != 'FRAMENO'})
    d['md'] = d.DEPTH / 1000.0
    d = d.drop(columns='DEPTH').replace(-999.25, np.nan)
    s = survey(well)
    d['tvd'] = np.interp(d.md, s.md, s.tvd)
    return d
