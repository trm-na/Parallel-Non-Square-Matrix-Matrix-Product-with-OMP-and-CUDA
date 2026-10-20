/**
 * @file matrix_mult_omp.c
 * @author trm_na
 * @brief CPU parallelisation of Matrix-Matrix multiplication using OMP.
 * 
 * Credits: The skeleton of this source code was developed by Salvatore Filippone.
 * 
 * @version 1.0
 * @date 2026-02-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */


/* Includes ------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

/* Matrix default dimensions -------------------------------------------------*/ 
const int DEF_N = 4096;
const int DEF_K = 128;


/**
 * @brief Matrix creation in row-major ordering.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param M Matrix pointer.
 * @param zeroes 0 = initialise matrix values with pseudo-random numbers | 1 = initialise matrix values with zeroes
 */
void CreateMatRows(int N, int K, float* M, int zeroes){
  int row, col, idx;
  for (row = 0; row < N; row++){
    for (col = 0; col < K; col++){
      idx = row * K + col;
      if(zeroes){
        M[idx] = 0.0;
      }else{
        M[idx] = rand() % 100 + 1;
      }
    }
  }
}

/**
 * @brief Matrix creation in column-major ordering.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param M Matrix pointer.
 * @param zeroes 0 = initialise matrix values with pseudo-random numbers | 1 = initialise matrix values with zeroes
 */
void CreateMatCols(int N, int K, float* M, int zeroes){
  int row, col, idx;
  for (col = 0; col < K; col++){
    for (row = 0; row < N; row++){
      idx = col * N + row;
      if(zeroes){
        M[idx] = 0.0;
      }else{
        M[idx] = rand() % 100 + 1;
      }
    }
  }
}

/**
 * @brief Simple CPU implementation of serial matrix multiplication A * B = C using row-major ordering.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
void SerialMatrixMultRows(int N, int K, const float* A, const float* B, float* C) {
  int rowc, colc, cola, idxa, idxb, idxc;
  for (rowc = 0; rowc < N; ++rowc) {
    for (colc = 0; colc < N; ++colc) {
      float t = 0.0;
      for (cola = 0; cola < K; ++cola) {
        idxa = rowc * K + cola;
        idxb = cola * N + colc;
        t += A[idxa] * B[idxb];
      }
      idxc = rowc * N + colc;
      C[idxc] = t;
    }
  }
}

/**
 * @brief Simple CPU implementation of serial matrix multiplication A * B = C using column-major ordering.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
void SerialMatrixMultCols(int N, int K, const float* A, const float* B, float* C) {
  int rowc, colc, cola, idxa, idxb, idxc;
  for (colc = 0; colc < N; ++colc) {
    for (rowc = 0; rowc < N; ++rowc) {
      float t = 0.0;
      for (cola = 0; cola < K; ++cola) {
        idxa = cola * N + rowc;
        idxb = colc * K + cola;
        t += A[idxa] * B[idxb];
      }
      idxc = colc * N + rowc;
      C[idxc] = t;
    }
  }
}

/**
 * @brief Simple CPU implementation of parallel matrix multiplication A * B = C using row-major ordering and OpenMP.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
void MatrixMultRows(int N, int K, const float* A, const float* B, float* restrict C) 
{
  int rowc,colc,rowb,cola,idxa,idxb,idxc,tid,nthreads,chunk;
  #pragma omp parallel shared(N, K, A, B, C, chunk) private(tid, rowc, colc, cola)
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    chunk = N/nthreads;
    #pragma omp for schedule(static,chunk) nowait
    for (rowc = 0; rowc < N; ++rowc) {
      for (colc = 0; colc < N; ++colc) {
        float t = 0.0;
          for (cola=0; cola < K; ++cola) {
            idxa = rowc * K + cola;
            idxb = cola * N + colc;
            t += A[idxa] * B[idxb];
          }
        idxc = rowc * N + colc;
        C[idxc] = t;
      }
    }
  }
}


/**
 * @brief Simple CPU implementation of parallel matrix multiplication A * B = C using column-major ordering.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
void MatrixMultCols(int N, int K, const float* A, const float* B, float* restrict C) 
{
  int rowc,colc,rowb,cola,idxa,idxb,idxc,tid,nthreads,chunk;
  #pragma omp parallel shared(N, K, A, B, C, chunk) private(tid, rowc, colc, cola)
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    chunk = N/nthreads;
    #pragma omp for schedule(static,chunk) nowait
    for (colc = 0; colc < N; ++colc) {
      for (rowc = 0; rowc < N; ++rowc) {
        float t = 0.0;
          for (cola=0; cola < K; ++cola) {
            idxa = cola * N + rowc;
            idxb = colc * K + cola;
            t += A[idxa] * B[idxb];
          }
        idxc = colc * N + rowc;
        C[idxc] = t;
      }
    }
  }
}

/**
 * @brief Obtain the maximum difference between the elements from row-major-ordered matrices C and D.
 * 
 * @param N Number of rows and columns.
 * @param C Matrix pointer.
 * @param D Matrix pointer.
 * @return float Maximum difference between C and D elements.
 */
float DiffRows(int N, const float* C, const float* D){
  float diff = 0.0f;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      int idx = row * N + col;
      diff = fmax(diff, fabs(D[idx] - C[idx]));
    }
  }
  return diff;
}

/**
 * @brief Obtain the maximum difference between the elements from column-major-ordered matrices C and D.
 * 
 * @param N Number of rows and columns.
 * @param C Matrix pointer.
 * @param D Matrix pointer.
 * @return float Maximum difference between C and D elements.
 */
float DiffCols(int N, const float* C, const float* D){
  float diff = 0.0f;
  for (int col = 0; col < N; ++col) {
    for (int row = 0; row < N; ++row) {
      int idx = col * N + row;
      diff = fmax(diff, fabs(D[idx] - C[idx]));
    }
  }
  return diff;
}

/**
 * @brief Computes a non-square matrix-matrix product in a serial and parallel way inside the CPU.
 * For the parallelisation, OpenMP is used.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could be
 * the number of rows of matrix A and number of columns of matrix B, the number of columns of matrix
 * A and number of rows of matrix B, the number of threads, the ordering mode and the printing mode).
 * @return int 
 */
int main(int argc, char** argv) 
{

  /* Sizes can be arguments from input */
  int N, K, mode, csv, nthreads;

  /* If there are input arguments, use them as the values to use */
  if(argc == 6){
    N = atoi(argv[1]);  /* Number of rows of matrix A and number of columns of matrix B */
    K = atoi(argv[2]);  /* Number of columns of matrix A and number of rows of matrix B */
    nthreads = atoi(argv[3]); /* Number of threads (for printing purposes) */
    mode = atoi(argv[4]); /* 0: Row-major matrices; 1: Column-major matrices */
    csv = atoi(argv[5]);  /* 0: Print a human-readable output; 1: Print a csv */
  }
  /* If there are none input arguments, use the default values */
  else{
    N = DEF_N;
    K = DEF_K;
    mode = 0;
    csv = 0;
  }

  /* Avoid using non-valid values for mode */
  if(mode < 0 || mode > 1){
    mode = 0;
  }

  /* Avoid using non-valid values for csv */
  if(csv < 0 || csv > 1){
    csv = 0;
  }

  /* Print metadata */
  if(csv){
    fprintf(stdout, "%i,%i,%i,", N, K, nthreads);    
  }else{
    fprintf(stdout, "N = %i, K = %i, number of threads = %i\n", N, K, nthreads);
  }


  
  /* -----------------------  Memory initialisation ----------------------- */
  
  /* Allocate the necessary memory for the matrices */
  float* A = (float*) malloc(sizeof(float)*N*K);
  float* B = (float*) malloc(sizeof(float)*K*N);
  float* C = (float*) malloc(sizeof(float)*N*N);
  float* D = (float*) malloc(sizeof(float)*N*N);

  /* Set a random seed to be used so that the results can be replicated */
  srand(42);

  /* Initialise A and B with random data and C and D with zeroes */
  if(!mode){
    CreateMatRows(N, K, A, 0);
    CreateMatRows(K, N, B, 0);
    CreateMatRows(N, N, C, 1);
    CreateMatRows(N, N, D, 1);
  }else{
    CreateMatCols(N, K, A, 0);
    CreateMatCols(K, N, B, 0);
    CreateMatCols(N, N, C, 1);
    CreateMatCols(N, N, D, 1);
  }

  /* Perform serial computations */
  double ser_tmlt;
  if(!mode){
    double t1 = omp_get_wtime();
    SerialMatrixMultRows(N, K, A, B, C);
    double t2 = omp_get_wtime();
    ser_tmlt = (t2-t1) * (1.0e3);
  }else{
    double t1 = omp_get_wtime();
    SerialMatrixMultCols(N, K, A, B, C);
    double t2 = omp_get_wtime();
    ser_tmlt = (t2-t1) * (1.0e3);
  }

  /* Compute serial MFLOPS */
  double ser_mflops = (2.0e-6)*N*N*K/ser_tmlt;

  /* Print serial results */
  if(csv){
    fprintf(stdout, "%lf,%lf,", ser_tmlt, ser_mflops);
  }else{
    fprintf(stdout, "Serial time: %lf ms. Serial MFLOPS: %lf\n", ser_tmlt, ser_mflops);
  }


  /* Perform parallel computations */
  double par_tmlt;
  if(!mode){
    double t1 = omp_get_wtime();
    MatrixMultRows(N, K, A, B, D);
    double t2 = omp_get_wtime();
    par_tmlt = (t2-t1) * (1.0e3);
  }else{
    double t1 = omp_get_wtime();
    MatrixMultCols(N, K, A, B, D);
    double t2 = omp_get_wtime();
    par_tmlt = (t2-t1) * (1.0e3);
  }

  /* Compute parallel MFLOPS */
  double par_mflops = (2.0e-6)*N*N*K/par_tmlt;

  /* Print parallel results */
  if(csv){
    fprintf(stdout, "%lf,%lf,", par_tmlt, par_mflops);
  }else{
    fprintf(stdout, "Parallel time: %lf ms. Parallel MFLOPS: %lf\n", par_tmlt, par_mflops);
  }

  /* Check if there are differences between the serial and parallel results */
  float diff;
  if(!mode){
    diff = DiffRows(N, D, C);
  }else{
    diff = DiffCols(N, D, C);
  }
  
  /* Print the maximum difference between serial and parallel results (Should be (very close to) zero) */
  if(csv){
    fprintf(stdout, "%f\n", diff);
  }else{
    fprintf(stdout, "Max diff = %f\n\n\n", diff);
  }


  // Uncomment for correctness tests (prints the solution matrix)
  //
  // if(!mode){
  //   printf("SERIAL SOLUTION:\n");
  //   for (int row = 0; row < N; ++row) {
  //     printf("|");
  //     for (int col = 0; col < N; ++col) {
  //       int idxr = row * N + col;
  //       fprintf(stdout, "%f ", C[idxr]);
  //     }
  //     fprintf(stdout, "|\n");
  //   }
  //   fprintf(stdout, "\n");
  //   fprintf(stdout, "\n");

  //   printf("PARALLEL SOLUTION:\n");
  //   for (int row = 0; row < N; ++row) {
  //     printf("|");
  //     for (int col = 0; col < N; ++col) {
  //       int idxr = row * N + col;
  //       fprintf(stdout, "%f ", D[idxr]);
  //     }
  //     fprintf(stdout, "|\n");
  //   }
  //   fprintf(stdout, "\n");
  //   fprintf(stdout, "\n");
  // }else{
  //   printf("SERIAL SOLUTION:\n");
  //   for (int col = 0; col < N; ++col) {
  //     printf("|");
  //     for (int row = 0; row < N; ++row) {
  //       int idxr = col * N + row;
  //       fprintf(stdout, "%f ", C[idxr]);
  //     }
  //     fprintf(stdout, "|\n");
  //   }
  //   fprintf(stdout, "\n");
  //   fprintf(stdout, "\n");

  //   printf("PARALLEL SOLUTION:\n");
  //   for (int col = 0; col < N; ++col) {
  //     printf("|");
  //     for (int row = 0; row < N; ++row) {
  //       int idxr = col * N + row;
  //       fprintf(stdout, "%f ", D[idxr]);
  //     }
  //     fprintf(stdout, "|\n");
  //   }
  //   fprintf(stdout, "\n");
  //   fprintf(stdout, "\n");
  // }

  /* Free matrices from memory */
  free(A);
  free(B);
  free(C);
  free(D);
  return 0;
}
