#!/bin/bash
set -Ceux

model_path=$1
data_root=$2

poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000152/clscale0.2/alpha5.0_conductivity1.000e-02_rep0 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w  $data_root/interim/part_2/00000152/clscale0.2/alpha5.0_conductivity1.000e-02_rep0

poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000164/clscale0.3/alpha5.0_conductivity1.000e-02_rep1 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w  $data_root/interim/part_2/00000164/clscale0.3/alpha5.0_conductivity1.000e-02_rep1

poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000164/clscale0.3/alpha5.0_conductivity1.000e-02_rep0 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w  $data_root/interim/part_2/00000164/clscale0.3/alpha5.0_conductivity1.000e-02_rep0

poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000126/clscale0.3/alpha1.0_conductivity1.000e-02_rep1 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w  $data_root/interim/part_2/00000126/clscale0.3/alpha1.0_conductivity1.000e-02_rep1

poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000086/clscale0.25/alpha1.0_conductivity1.000e-02_rep2 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w  $data_root/interim/part_1/00000086/clscale0.25/alpha1.0_conductivity1.000e-02_rep2



poetry run python3 src/infer.py \
  $model_path \
  $data_root/preprocessed/test_50/00000126/clscale0.2/alpha1.0_conductivity1.000e-02_rep1 \
  -p $data_root/preprocessed/preprocessors.pkl \
  -w $data_root/interim/part_2/00000126/clscale0.2/alpha1.0_conductivity1.000e-02_rep1

