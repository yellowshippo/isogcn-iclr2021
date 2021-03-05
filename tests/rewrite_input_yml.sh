#!/bin/bash
set -Ceux

if [ $# -ne 3 ]
then
  echo "Usage: $0 original_yaml scale_yaml key"
  exit 1
fi

original_yaml=$1
scale_yaml=$2
key=$3

scale_value=$(grep $key $scale_yaml | grep -oE [0-9]\.[0-9]+[eE]?[+-]?[0-9]*)
sed "s/factor: .*/factor: $scale_value/" $original_yaml > "${original_yaml}.scaled.yml"

exit 0
