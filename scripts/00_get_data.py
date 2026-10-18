"""Download the open FORCE 2020 training wells and the penalty matrix into ./data.

Source: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
Citation: Bormann P., Aursand P., Dilib F., Dischington P., Manral S. (2020), doi:10.5281/zenodo.4351156
Check the repository for its licence terms before reusing the data.
"""
import os
import urllib.request
import zipfile

from paths import DATA

BASE = 'https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/'

for name in ('train.zip', 'penalty_matrix.npy'):
    target = DATA + name
    if not os.path.exists(target):
        print('downloading', name)
        urllib.request.urlretrieve(BASE + name, target)

if not os.path.exists(DATA + 'train.csv'):
    with zipfile.ZipFile(DATA + 'train.zip') as z:
        z.extractall(DATA)
print('data ready in', DATA)
