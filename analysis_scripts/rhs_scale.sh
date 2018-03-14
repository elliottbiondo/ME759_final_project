#!/bin/bash

for k in cuda mkl
do
  for j in 1 5 20 101 301 501 601 801 999 1200;
  do
    for i in `seq 1 10`;
    do
      ../${k}_solve 100 3 $j  | cat >> rhs_res_${k}
    done
  done
done
