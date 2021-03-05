#!/bin/bash

set -Ceux

if [ $# -eq 2 ]
then
  msh_file_name=$2
else
  msh_file_name='heat.msh'
fi

find "$1" -name "$msh_file_name" | while read file
do
  directory="$(dirname $file)"
  pushd "$directory"
  OMP_NUM_THREADS=1 fistr1
  popd
done
