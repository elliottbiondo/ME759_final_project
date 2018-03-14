#!/bin/bash

for k in cuda mkl
do
  for j in 5 20 101 201 301 401 501 601 701 801 901 999 1200 1400 1600 2000 3000 4000 5000;
  do
    for i in `seq 1 10`;
    do
      ../${k}_solve $j 3 1 | cat >> dimension_res_${k}
    done
  done
done
