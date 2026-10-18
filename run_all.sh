#!/usr/bin/env bash
# Runs the open-data experiments in order. The first three take a while on a laptop.
set -e
cd "$(dirname "$0")/scripts"
python 00_get_data.py
python 01_lithology.py
python 02_reconstruction.py
python 03_bilstm_dtc.py
python 04_geomechanics_open_wells.py
python 06_make_figures.py
python 07_extra_figures.py
# 05_field_case.py needs the proprietary mud-log workbook:  FIELD_XLSX=/path/to/file.xlsx python 05_field_case.py
