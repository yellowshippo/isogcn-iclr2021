#!/bin/bash
set -Ceux

model_path=$1

poetry run python3 src/infer.py $model_path \
  data/std/preprocessed/validation_50/00000133/clscale0.3/alpha10.0_conductivity1.000e-02_rep0/ \
  -w data/interim/part_2/00000133/clscale0.3/alpha10.0_conductivity1.000e-02_rep0/

poetry run python3 src/infer.py $model_path \
  data/std/preprocessed/validation_50/00000133/clscale0.3/alpha1.0_conductivity1.000e-02_rep0/ \
  -w data/interim/part_2/00000133/clscale0.3/alpha1.0_conductivity1.000e-02_rep0/

poetry run python3 src/infer.py $model_path \
  data/std/preprocessed/validation_50/00000133/clscale0.2/alpha10.0_conductivity1.000e-02_rep0/ \
  -w data/interim/part_2/00000133/clscale0.2/alpha10.0_conductivity1.000e-02_rep0/

poetry run python3 src/infer.py $model_path \
  data/std/preprocessed/validation_50/00000133/clscale0.2/alpha1.0_conductivity1.000e-02_rep0/ \
  -w data/interim/part_2/00000133/clscale0.2/alpha1.0_conductivity1.000e-02_rep0/

poetry run python3 src/infer_raw.py $model_path \
  tests/data/mesh_size/raw/alpha10.0_conductivity1.000e-02
