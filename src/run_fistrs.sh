#!/bin/bash
set -x

if [ $# -lt 2 ]
then
  echo "Usage: $0 root_directory n_run [options]"
  exit 0
fi

root_directory=$1
n_run=$2
options=${@:3}
echo $options

for i in $(seq 1 $n_run)
do
  qexe $options $(dirname $0)/run_fistr.sh $root_directory $i $n_run
done
