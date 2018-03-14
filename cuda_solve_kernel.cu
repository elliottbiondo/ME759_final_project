#include <stdio.h>
#include "cuda_solve.h"

__global__ void SetupMult(double* A, int n, int k, int half_k, int i){
  int tid = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  __shared__ double diag;

  if(threadIdx.y==0){
    diag = A[i*k + half_k];
  }
  __syncthreads();

  if(tid < half_k and i+1+tid < n)
    A[(i+1+tid)*k + half_k - 1 - tid] = A[(i+1+tid)*k + half_k - 1 - tid]/diag;
}

__global__ void LUDecomp(double* A, int n, int k, int half_k, int i){

  int tidx = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  int tidy = blockIdx.y*BLOCK_SIZE + threadIdx.y;

  int row = tidy + i + 1;
  int col = tidx + half_k - tidy;

  if(tidx<half_k && tidy<half_k && row<n && col < k){
    A[row*k + col] = A[row*k + col] - A[row*k + col - tidx - 1]*A[i*k + half_k + tidx + 1];
  }

}

__global__ void ForwardSolve(double* A, double* b, int n, int k, int half_k, int i){
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int tidy = by*BLOCK_SIZE2+ty;
  int row = tidy + i + 1;
  __shared__ double mult;

  if(ty==0){
    mult = b[i];
  }

  __syncthreads();

  if(tidy < half_k && row < n){
    b[row] = b[row] - A[row*k + half_k - 1 - tidy]*mult;
  }

}

__global__ void BackSolve(double* A, double* b, int n, int k, int half_k, int i){
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int tidy = by*BLOCK_SIZE2+ty;
  int row = i - 1 - tidy;
  __shared__ double mult;

  if(ty==0){
    b[i] = b[i]/A[i*k + half_k];
    mult = b[i];
  }

  __syncthreads();

  if(tidy < half_k && row >= 0){
    b[row] = b[row] - A[row*k + half_k + 1 + tidy]*mult;
  }

}
