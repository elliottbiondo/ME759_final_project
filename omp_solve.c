#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

void fillA(double* A, int n, int k);
void fillb(double* b, double* A, int n, int k, int w);
void OmpSolve(double*A, double*b, int n, int k, int w);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv){

  int n;
  int k;
  int j;
  int w; //width of b

  switch(argc-1){
  case 3:
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    w = atoi(argv[3]);
    srand(119);
  break;
  default:
    printf("\nSupply exactly 3 arguments: See README_BUILD_AND_USAGE\n");
    exit(1);
  break;
  } 

  //allocate matricies
  double *A, *b;
  A = (double *) malloc(n*k*sizeof(double));
  b = (double *) malloc(n*w*sizeof(double));


 //fill matricies with either random or supplied values
  switch(argc-1){
    case 3:
      fillA(A, n, k);
      fillb(b, A, n, k, w);
    break;
  }
  //solve the system of equations
  OmpSolve(A,b,n,k,w);
  //If running with random matricies, test results;
  //If running with supplied matricies, print output to stdout
  switch(argc-1){
    case 3:
      for(j=0; j<n*w; j++){ 
          if(fabs(b[j] - 1.0000) > 0.001f || b[j] != b[j]){
            printf("\nFailed at x[%i] = %f", j, b[j]);
            exit(1);
          }
      }
      printf("Test PASSED");
    break;
  }

   // Free matrices
  free(A);
  free(b);
  return 0;
}

void fillb(double* b, double* A, int n, int k, int w){
  int i, j, l;
  for(l=0; l<w; l++){
    for(i=0; i<n; i++){
      for(j=0; j<k; j++){
        b[i + n*l] += A[i*k + j];
      }
    }   
  }
}

//Created matrix garanteed to be diagonally dominent
void fillA(double* A, int n, int k){
   int half_k = (k-1)/2; //convert k from a bandwidth to a half bandwidth
   int i, j;
   for(i=0; i<n; i++){
     for(j=0; j<k; j++){
       if(j == half_k)
         A[i*k + j] = (double) rand()/RAND_MAX + (double) k;
       else if (j > half_k - 1 - i &&  j < half_k + n - i)
         A[i*k + j] = (double) rand()/RAND_MAX;
       else
         A[i*k + j] = 0.0;
     }
   }

}

void OmpSolve(double *A, double *b, int n, int k, int w){
  int i,j,l;
  int half_k = (k-1)/2;
  int diag;
  double startTime, endTime, elapsedTime;
  int maxThreads = omp_get_max_threads();
  printf("Performing substitution with num thread: %i\n", maxThreads);
  omp_set_num_threads(maxThreads);

  startTime = omp_get_wtime();
  //preform LU decomposition

  for(i=0; i<n-1; i++){
    for(j=i+1; j<half_k+i+1; j++){
      if(j<n) A[j*k + half_k - 1 - (j-i-1)] /= A[i*k + half_k];
    }
    for(j=i+1; j<half_k+i+1; j++){
      for(l=i+1; l<half_k+i+1; l++){
        if(j<n){
        if(i<n) A[j*k+half_k+(l-i-1)-(j-i-1)] -= A[j*k+half_k-1-(j-i-1)]*A[i*k+half_k+1+(l-i-1)];
        }
      }
    }
  }

  #pragma omp parallel for private(i,j)
  for(l=0; l<w; l++){
    for(i=0; i<n; i++){
      for(j=i+1; j<half_k+i+1; j++){
        if(j<n) b[j+n*l] -= b[i+n*l]*A[j*k + half_k - 1 - (j-i-1)];
      }
    }
    
    for(i=n-1; i>=0; i--){
      b[i+n*l] = b[i+n*l]/A[i*k + half_k];
      for(j=i-1; j>i-1-half_k; j--){
        if(j>=0) b[j+n*l] -= b[i+n*l]*A[j*k+half_k+1+(i-1-j)];
      }
    }
  }

  endTime = omp_get_wtime();
  elapsedTime = (endTime - startTime)*1000; //convert s to ms
  printf("Execution time(ms): %f\n", elapsedTime);

}

