#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: $0 model_root_directory data_directory preprocessors.pkl"
  exit 1
fi

set -Ceux

model_root_directory=$1
data_directory=$2
preprocessors_pkl=$3

find -L $model_root_directory -name settings.yml | while read setting_file
do
  model_directory=$(dirname $setting_file)
  mkdir -p ${data_directory}/${model_directory}
  poetry run python3 src/infer.py \
    $model_directory \
    $data_directory \
    -p $preprocessors_pkl \
    -l ${data_directory}/${model_directory} &
done

exit 0
