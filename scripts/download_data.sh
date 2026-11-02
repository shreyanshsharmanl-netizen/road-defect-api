#!/usr/bin/env bash
# CrackForest is for non-commercial research use and is not stored in this repo.
set -euo pipefail
mkdir -p data
[ -d data/crackforest ] || git clone --depth 1 https://github.com/cuilimeng/CrackForest-dataset data/crackforest
