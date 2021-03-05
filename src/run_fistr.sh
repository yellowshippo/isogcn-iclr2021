#!/bin/bash
set -x

root_directory=$1
i_run=$2
n_run=$3

for f in $(find $root_directory -name heat.msh | sed -ne "${i_run}~${n_run}p")
do
  if [ -f $(dirname $f)/heat.res.0.100 ]
  then
    echo "Already computed. Skip: $(dirname $f)"
    continue
  fi

  pushd $(dirname $f)
  fistr1
  popd
done
