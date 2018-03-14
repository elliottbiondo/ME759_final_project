all: cuda mkl omp serial ompslow
cuda:
	nvcc -O3 -arch=sm_20 -o cuda_solve cuda_solve.cu cuda_solve_kernel.cu
mkl:
	$(ICC) -O3 -mkl -o mkl_solve mkl_solve.c
omp:
	gcc -O3 -fopenmp -o omp_solve omp_solve.c
serial:
	gcc -O3 -fopenmp -o serial_solve serial_solve.c
ompslow:
	gcc -O3 -fopenmp -o omp_solve_slow omp_solve_slow.c
clean:
	rm -rf cuda_solve mkl_solve omp_solve serial_solve omp_solve_slow
