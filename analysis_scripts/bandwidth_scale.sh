#!/bin/bash

for k in cuda mkl
do
  for j in 101 201 301 401 501 601 701 801 901 999;
  do
    for i in `seq 1 10`;
    do
      ../${k}_solve 1000 $j 1 | cat >> bandwidth_res_${k}
    done
  done
done
