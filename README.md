# ai-wellbore-twin

Code and saved results for the manuscript *An Open-Data Benchmark and Field Case for AI-Assisted Lithology, Rock-Property and Pressure Estimation from Drilling, Log and Mud-Gas Data* (R. Ali).

The question behind the code is a practical one: what can learned models and classical relations tell a wellsite team about lithology, rock properties, pressure and gas character when only part of the usual data exists? Every number in the paper's main experiments comes from an open data set, validated with whole wells held out.

## Data

The open wells are the training set of the FORCE 2020 lithology competition (98 North Sea wells).

* Repository: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
* Citation: Bormann P., Aursand P., Dilib F., Dischington P., Manral S. (2020). doi:10.5281/zenodo.4351156
* Check the repository for its licence terms before reusing the data.

`scripts/00_get_data.py` downloads what is needed into `data/` (not tracked by git).

The onshore field record used in the paper's field case belongs to the well operator. It is **not** in this repository, and neither are outputs derived from it. `scripts/05_field_case.py` shows the method and runs on any workbook with the same sheet layout.

## Running it

```
pip install -r requirements.txt
./run_all.sh
```

Scripts and what they do:

| Script | Purpose |
|---|---|
| `00_get_data.py` | fetch the FORCE 2020 training wells and penalty matrix |
| `01_lithology.py` | threshold rules, Random Forest, XGBoost, SVM; five folds split by well |
| `02_reconstruction.py` | bulk density and shear slowness from other logs; Poisson's ratio |
| `03_bilstm_dtc.py` | Bi-LSTM versus point-wise XGBoost for compressional slowness |
| `04_geomechanics_open_wells.py` | overburden, Eaton pore pressure and fracture gradient on eligible wells |
| `05_field_case.py` | d-exponent and Haworth / Pixler gas ratios (needs the private workbook) |
| `06_make_figures.py` | figures from the saved results |
| `07_extra_figures.py` | lithology diagnostics and per-well geomechanics figures |

Expect a few hours for the first three on a two-core machine. The large intermediate arrays are re-created by the scripts and are not stored here.

## Main results

Lithology, five folds, wells held out, pooled over test folds:

| Model | Accuracy | Balanced acc. | Macro F1 | FORCE score |
|---|---|---|---|---|
| Empirical thresholds | 0.378 | 0.236 | 0.238 | -1.71 |
| Random Forest | 0.701 | 0.415 | 0.427 | -0.78 |
| XGBoost | 0.710 | 0.321 | 0.345 | -0.76 |
| SVM (RBF) | 0.527 | 0.384 | 0.304 | -1.23 |

Reconstruction (R squared, pooled over wells held out): density 0.78 (XGBoost), shear slowness 0.90 (MLP); Poisson's ratio from predicted shear slowness, MAE 0.026. Bi-LSTM R squared for DTC 0.90 against 0.90 for XGBoost, so no gain.

The pressure chain has no ground truth in the open data. Results in `results/geomech_perwell.csv` are plausibility checks against the mud weight used.

## Layout

```
scripts/   numbered analysis scripts and paths.py
results/   metrics (json/csv) and figures built from open data
data/      downloaded FORCE files (ignored by git)
private/   place for anything derived from proprietary data (ignored by git)
```

## Caveats

* Lithology labels are expert interpretations, and depth is used as an input, so the lithology scores describe this basin, not every basin.
* Rare lithologies (dolomite, anhydrite, basement) are not learned.
* The Eaton step needs a usable normal compaction trend; one was found in only three of eight eligible wells.
* This is an offline analysis. It has not been run on a live data stream.

## Licence

Code: MIT (see LICENSE). The FORCE data keep their own terms.
