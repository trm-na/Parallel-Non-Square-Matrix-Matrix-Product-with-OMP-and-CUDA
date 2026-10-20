/**
 * @file matrix_mult_cuda.c
 * @author trm_na
 * @brief CPU parallelisation of Matrix-Matrix multiplication using OMP.
 * 
 * Credits: The skeleton of this source code was developed by Michał Czapiński in 2014.
 * 
 * @version 1.0
 * @date 2026-02-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

 
/* Includes ------------------------------------------------------------------*/
#include <iostream>

#include <cuda_runtime.h>   // For CUDA runtime API
#include <helper_cuda.h>    // For checkCudaError macro
#include <helper_timer.h>   // For CUDA SDK timers

/* Matrix default dimensions -------------------------------------------------*/ 
const int DEF_N = 4096;
const int DEF_K = 128;

/* Block dimension -----------------------------------------------------------*/ 
const dim3 BLOCK_DIM(16, 16);

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
        M[idx] = std::rand() % 100 + 1;
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
        M[idx] = std::rand() % 100 + 1;
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
void CpuMatrixMultRows(int N, int K, const float* A, const float* B, float* C) {
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
void CpuMatrixMultCols(int N, int K, const float* A, const float* B, float* C) {
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
 * @brief GPU implementation of parallel matrix multiplication A * B = C using row-major ordering and one CUDA thread per vector element.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
__global__ void GpuMatrixMultRows(int N, int K, const float* A, const float* B, float* C) {
  /* Calculate indices of matrix elements multiplied by this thread. Assume 2D grid of blocks */
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y; 

  /* Calculate the element index in the global memory and add the values */
  /* No threads access memory outside the allocated area */
  if (row < N && col < N) {
    float t = 0.0;
    for(int i = 0; i < K; ++i){
      int idxa = row * K + i;
      int idxb = col + i * N;
      t += A[idxa] * B[idxb];
    }
    int idxc = row * N + col;
    C[idxc] = t;
  }
}

/**
 * @brief GPU implementation of parallel matrix multiplication A * B = C using column-major ordering and one CUDA thread per vector element.
 * 
 * @param N Number of rows.
 * @param K Number of columns.
 * @param A Matrix A pointer
 * @param B Matrix B pointer.
 * @param C Matrix C pointer.
 */
__global__ void GpuMatrixMultCols(int N, int K, const float* A, const float* B, float* C) {
  /* Calculate indices of matrix elements multiplied by this thread. Assume 2D grid of blocks */
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y; 

  /* Calculate the element index in the global memory and add the values */
  /* No threads access memory outside the allocated area */
  if (row < N && col < N) {
    float t = 0.0;
    for(int i = 0; i < K; ++i){
      int idxa = i * N + row;
      int idxb = col * K + i;
      t += A[idxa] * B[idxb];
    }
    int idxc = col * N + row;
    C[idxc] = t;
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
      diff = std::max(diff, std::abs(D[idx] - C[idx]));
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
      diff = std::max(diff, std::abs(D[idx] - C[idx]));
    }
  }
  return diff;
}

/**
 * @brief Computes a non-square matrix-matrix product in a serial way inside the CPU and in a 
 * parallel one inside the GPU.For the parallelisation, CUDA is used.
 * 
 * @param argc Number of arguments passed to the programme in the command line.
 * @param argv Arguments passed to the programme in the command line (in this programme, it could be
 * the number of rows of matrix A and number of columns of matrix B, the number of columns of matrix
 * A and number of rows of matrix B, the ordering mode and the printing mode).
 * @return int 
 */
int main(int argc, char** argv) {

  /* Sizes can be arguments from input */
  int N, K, mode, csv;

  /* If there are input arguments, use them as the values to use */
  if(argc == 5){
    N = atoi(argv[1]);  /* Number of rows of matrix A and number of columns of matrix B */
    K = atoi(argv[2]);  /* Number of columns of matrix A and number of rows of matrix B */
    mode = atoi(argv[3]); /* 0: Row-major matrices; 1: Column-major matrices */
    csv = atoi(argv[4]);  /* 0: Print a human-readable output; 1: Print a csv */
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
    std::cout << N << "," << K << ",";
  }else{
    std::cout << "N = " << N << ", K = " << K << std::endl << std::endl;
  }

// ----------------------- Host memory initialisation ----------------------- //

  /* Allocate the necessary memory for the host (CPU) matrices */
  float* h_A = new float[N * K];
  float* h_B = new float[K * N];
  float* h_C = new float[N * N];
  float* h_D = new float[N * N];

  /* Set a random seed to be used so that the results can be replicated */
  std::srand(42);

  /* Initialise A and B with random data and C and D with zeroes */
  if(!mode){
    CreateMatRows(N, K, h_A, 0);
    CreateMatRows(K, N, h_B, 0);
    CreateMatRows(N, N, h_C, 1);
    CreateMatRows(N, N, h_D, 1);
  }else{
    CreateMatCols(N, K, h_A, 0);
    CreateMatCols(K, N, h_B, 0);
    CreateMatCols(N, N, h_C, 1);
    CreateMatCols(N, N, h_D, 1);
  }  


// ---------------------- Device memory initialisation ---------------------- //

  /* Allocate the global memory for the device (GPU) matrices */
  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaMalloc(&d_A, N * K * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_B, K * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_C, N * N * sizeof(float)));

  /* Create the CUDA SDK timer */ 
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  /* Copy matrices from the host (CPU) to the device (GPU) */
  timer->start();
  checkCudaErrors(cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice));
  timer->stop();

  /* Get the communication time */
  float gpu_comm_time = timer->getTime();

  /* Print the communication time */
  if(csv){
    std::cout << gpu_comm_time << ",";
  }else{
    std::cout << "GPU communication time: " << gpu_comm_time << " ms." << std::endl;
  }


// ------------------------ Calculations on the CPU ------------------------- //

  /* Perform serial computations */
  if(!mode){
    timer->reset();
    timer->start();
    CpuMatrixMultRows(N, K, h_A, h_B, h_C);
    timer->stop();
  }else{
    timer->reset();
    timer->start();
    CpuMatrixMultCols(N, K, h_A, h_B, h_C);
    timer->stop();
  }

  /* Get the CPU time and compute its associated MFLOPS */
  float cpu_time = timer->getTime();
  float cpu_mflops = (2.0e-6)*N*N*K/cpu_time;

  /* Print CPU results */
  if(csv){
    std::cout << cpu_time << "," << cpu_mflops << ",";
  }else{
    std::cout << "CPU time: " << cpu_time << " ms. CPU MFLOPS: " << cpu_mflops << std::endl;
  }

// ------------------------ Calculations on the GPU ------------------------- //

  /* Calculate the dimension of the grid of blocks (2D) */
  const dim3 GRID_DIM((N - 1)/BLOCK_DIM.x + 1, (N - 1)/BLOCK_DIM.y + 1);

  /* Perform parallel computations */
  if(!mode){
    timer->reset();
    timer->start();
    GpuMatrixMultRows<<<GRID_DIM, BLOCK_DIM>>>(N, K, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
  }else{
    timer->reset();
    timer->start();
    GpuMatrixMultCols<<<GRID_DIM, BLOCK_DIM>>>(N, K, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
  }

  /* Get the GPU time and compute its associated MFLOPS */
  float gpu_time = timer->getTime();
  float gpu_mflops = (2.0e-6)*N*N*K/gpu_time;

  /* Print GPU results */
  if(csv){
    std::cout << gpu_time << "," << gpu_mflops << ",";
  }else{
    std::cout << "GPU time: " << gpu_time << " ms. GPU MFLOPS: " << gpu_mflops << std::endl;
  }

  /* Download the resulting matrix d_C from the device and store it in h_D */
  checkCudaErrors(cudaMemcpy(h_D, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  /* Check if there are differences between the serial and parallel results */
  float diff;
  if (!mode){
    diff = DiffRows(N, h_D, h_C);
  }else{
    diff = DiffCols(N, h_D, h_C);
  }
  
  /* Print the maximum difference between serial and parallel results (Should be (very close to) zero) */
  if(csv){
    std::cout << diff << std::endl;
  }else{
    std::cout << "Max diff = " << diff << std::endl << std::endl << std::endl;
  }


// ------------------------------- Correctness tests ------------------------------ //

  // Uncomment for correctness tests (prints the solution matrix)
  //
  // if(!mode){
  //   std::cout << "CPU SOLUTION:" << std::endl;
  //   for (int row = 0; row < N; ++row) {
  //     std::cout << "|";
  //     for (int col = 0; col < N; ++col) {
  //       int idxr = row * N + col;
  //       std::cout << h_C[idxr] << " ";
  //     }
  //     std::cout << "|" << std::endl;
  //   }
  //   std::cout << std::endl << std::endl;

  //   std::cout << "GPU SOLUTION:" << std::endl;
  //   for (int row = 0; row < N; ++row) {
  //     std::cout << "|";
  //     for (int col = 0; col < N; ++col) {
  //       int idxr = row * N + col;
  //       std::cout << h_D[idxr] << " ";
  //     }
  //     std::cout << "|" << std::endl;
  //   }
  //   std::cout << std::endl << std::endl;
  // }else{
  //   std::cout << "CPU SOLUTION:" << std::endl;
  //   for (int col = 0; col < N; ++col) {
  //     std::cout << "|";
  //     for (int row = 0; row < N; ++row) {
  //       int idxr = col * N + row;
  //       std::cout << h_C[idxr] << " ";
  //     }
  //     std::cout << "|" << std::endl;
  //   }
  //   std::cout << std::endl << std::endl;

  //   std::cout << "GPU SOLUTION:" << std::endl;
  //   for (int col = 0; col < N; ++col) {
  //     std::cout << "|";
  //     for (int row = 0; row < N; ++row) {
  //       int idxr = col * N + row;
  //       std::cout << h_D[idxr] << " ";
  //     }
  //     std::cout << "|" << std::endl;
  //   }
  //   std::cout << std::endl << std::endl;
  // }

// ------------------------------- Cleaning up ------------------------------ //

  /* Delete the timer */
  delete timer;

  /* Free matrices from GPU memory */
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  /* Free matrices from CPU memory */
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;
  return 0;
}
