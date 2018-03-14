#include <stdio.h>
#include "cuda_solve.h"

__global__ void LUDecomp(double* A, int n, int k, int half_k, int i);
__global__ void SetupMult(double* A, int n, int k, int half_k, int i);
__global__ void ForwardSolve(double* A, double* b, int n, int k, int half_k, int i);
__global__ void BackSolve(double* A, double* b, int n, int k, int half_k, int i);

