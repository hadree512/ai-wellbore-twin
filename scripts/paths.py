"""Folder layout used by every script. Everything is relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data') + os.sep          # open FORCE 2020 files (see 00_get_data.py)
RESULTS = os.path.join(ROOT, 'results') + os.sep    # metrics, saved predictions, figures
PRIVATE = os.path.join(ROOT, 'private') + os.sep    # anything derived from the proprietary field record
for d in (DATA, RESULTS, os.path.join(RESULTS, 'figs'), PRIVATE):
    os.makedirs(d, exist_ok=True)
