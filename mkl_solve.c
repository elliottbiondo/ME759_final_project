/*
This program creates a random, diagonally dominate band matrix to be solved by
dgbsv. The form of the matrix is consistent with the Intel MKL LAPACKE_?gbsv
documentation. It is in a band form that is different than what is used in the
rest of this project. Note that random values appear even in places that the
algorithm does not touch. Also note that unlike all of the other programs in
this progect, the solution matrix (x) is not all ones because the fillAB is not
as sophisticated (due to the difficulties associated with the different band
form).
*/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

void fillAB(double* A, double* B, int n, int k, int w);

int main(int argc, char* argv[])
{

  double *A, *B;
  int *ipiv;
  int N, w, k, i, j;
  struct timeval start, end;
  srand(119);

  switch(argc-1){
    case 3:
      N = atoi(argv[1]);
      k = atoi(argv[2]);
      w = atoi(argv[3]);
    break;
    default:
      printf("\nSupply exactly 3 arguments: matrix dim, bandwidth, number of right hand sides.\n");
      exit(1);
    break;
  } 

  //ensure k is odd
  if(k%2 ==0){
    printf("\nBandwidth must be odd\n");
    exit(1);
  }
              
  int half_k = (k-1)/2;
  A = (double *)mkl_malloc( N*(k+half_k)*sizeof( double ), 64 );
  B = (double *)mkl_malloc( N*w*sizeof( double ), 64 );
  ipiv = (int*)mkl_malloc( N*sizeof( int ), 32 );

  fillAB(A, B, N, k, w);

  gettimeofday(&start, NULL);
  LAPACKE_dgbsv(LAPACK_ROW_MAJOR, N, half_k, half_k, w, &(A[0]), N, &(ipiv[0]), &(B[0]), w);
  gettimeofday(&end, NULL);

  printf ("MKL time %f ms\n", (double) (end.tv_usec - start.tv_usec)/1000.0 + (double) (end.tv_sec - start.tv_sec)*1000.0);

 mkl_free(A);
 mkl_free(B);
 mkl_free(ipiv);

 return 0;
}

void fillAB(double* A, double* B, int N, int k, int w){
  int i, j;
  int half_k = (k-1)/2;

  for(i=0; i<k+half_k; i++){
    for(j=0; j<N; j++){
      if(i==2*half_k){
        A[i*N + j] = (double) rand()/RAND_MAX + (double) k;
      }
      else{
        A[i*N + j] = (double) rand()/RAND_MAX;
      }
    } 
  }

  for(i=0; i<N*w; i++){
    B[i] = (double) rand()/RAND_MAX;
  }

}


