#!/bin/bash

if [ $# -ne 1 ]
then
  echo "Usage: $0 fistr_data_root_directory"
  exit 1
fi

n_cpu=$(lscpu -p | grep '^[0-9]' | wc -l)

find $1 -name hecmw_ctrl.dat | xargs --max-procs=$n_cpu -L 1 bash -c 'cd "$(dirname $0)" && pwd && fistr1'

exit 0
