#!/bin/bash
set -Ceux

result=$(cut -d, -f3 tests/data/optimization/output/trials.csv | grep -v value | sort -n | head -1)
if (( $(echo "$result < 0.01" | bc -l) ))
then
  echo "Results OK: ${result}"
else
  echo "Results NOK: not smaller than 0.01 ${result}"
  exit 1
fi

