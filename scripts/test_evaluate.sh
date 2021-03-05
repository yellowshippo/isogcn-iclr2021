#!/bin/bash
set -Ceux

# NOTE: data/std_all is different (it uses mean volume instead of effective volume)
test_data_directory=data/adj_std_all/preprocessed/test_50
pkl=data/adj_std_all/preprocessed/preprocessors.pkl

out_base_directory=data/inferred/test_evaluate/$(date +%Y%m%d_%H%M%S)

cat scripts/model_directories.txt | while read directory
do
  output_directory=${out_base_directory}/$(basename $directory)
  qexe -c 12 -m 250 poetry run python3 src/infer.py $directory $test_data_directory -p $pkl -l $output_directory
done
