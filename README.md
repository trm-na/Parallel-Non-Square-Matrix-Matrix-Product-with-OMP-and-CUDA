# Intel MPI Parallel Sparse-Matrix Fat-Vector Multiplication

This repository includes the code to parallelise a non-square matrix-matrix multiplication using a shared memory architecture. The parallelisation will be performed in a CPU, using up to 16 cores, and a GPU to test which parallelisation method is more efficient. To do so, OpenMP and NVIDIA CUDA will be used respectively.

To describe the implemented methods, for illustration purposes, 4 x 3 and 3 x 4 matrices will be used, but these dimensions differ to the ones used for the computations. To refer to the real dimensions in a general way, matrix A is an N x K matrix and matrix B a K x N matrix, meaning that the result matrix, matrix C, will be an N x N matrix. Additionally, the number of threads will be illustrated as 3, whereas this number will vary throughout the analysis, and it will be referenced as nthreads.

## Serial algorithm

First, the problem was approached serially.

The serial solution for this problem is the regular matrix multiplication algorithm: the i<sup>th</sup> row of matrix A is multiplied by the j<sup>th</sup> column of matrix B, and the result of that multiplication will be the element C<sub>i,j</sub> of the solution matrix C.


The previous image shows how the serial algorithm works to obtain the first element of the solution matrix, where $C_{1,1} = A_{1,1} \cdot B_{1,1} + A_{1,2} \cdot B_{2,1} + A_{1,3} \cdot B_{3,1}$. This serial algorithm would go on, first computing all the elements from the first row of the solution matrix and then continuing in the same way for the subsequent rows. Additionally, as it is a serial algorithm, all the computation would be done by the same CPU thread.

Once the serial algorithm was determined, its parallelisation was designed. Even though the algorithm is essentially the same, as CPU parallelisation behaves differently than the GPU one using OpenMP (OMP) and CUDA respectively, the implementation is different depending on the device.

## CPU parallelisation (OpenMP)

As with OpenMP the number of threads can be obtained, the parallelisation algorithm was designed to balance the load, having each thread processing a block of rows from C to obtain the result.



The previous image depicts how the CPU parallel algorithm would work with 3 threads and the example matrices. Thread 1 has rows 1 and 2 assigned, thread 2 has row 3 and thread 3 has row 4. Each one of the threads accesses the shared memory to get and multiply the elements they need to compute the results they are assigned to obtain. In this case, each thread must access one (or two) rows from A and the entirety of B.

With the real dimensions, to balance the load, each thread processes N/nthreads rows from C, having some of them to process additional rows if N is not divisible by nthreads.

## GPU parallelisation (CUDA)

With CUDA, the parallelisation works differently. Instead of having a set number of threads, the threads act according to a block on the matrix to process.



The previous image shows how the GPU parallel algorithm would work with three random elements from matrix C. Each thread first determines its location in matrix C. Depending on its location, the thread will access the needed data from memory to compute the corresponding result. The threads multiply their corresponding row from A by their corresponding column from B and obtain the result for their location.

## Correctness tests

Additionally, some minor correctness tests were developed performed to determine if the CPU and the GPU give the same solution by printing the solution matrices for a modest problem. These tests are included in both OMP and CUDA files as a comment, as well as the submission files for their execution.

## Code execution

This code was tested and executed in Cranfield University’s Crescent2 (CRanfield Educational Scientific Computing ENvironment for Teaching), a supercomputer running on Linux with Intel Xeon E5 2620 CPUs, having 16 CPU cores per node, and an Nvidia V100 32GB GPU.

The PBS submission scripts (`cuda.sub`, `cuda_correctness.sub`, `openmp.sub` and `openmp_correctness.sub`) were adapted from Cranfield University's templates for CR2. The configuration is specific to CR2 and may require changes on other systems, particularly regarding:

- PBS queue and resource requests
- Available CPU cores
- GPU type and availability
- GCC, CUDA and CMake module versions
- Environment module paths
- Compiler and OpenMP configuration
- Executable location
- Input and output paths

To execute the code in CR2, first upload the CUDA and OMP directories. After that, execute `module use /apps/modules/all/`, `module load GCC/11.3.0 CUDA/11.7.0 CMake/3.24.3-GCCcore-11.3.0`, `export CC=$(which gcc)` and `export CXX=$(which g++)`.

To compile the CUDA files, execute `wget https://github.com/NVIDIA/cuda-samples/archive/v11.6.tar.gz` and `tar xzf v11.6.tar.gz`. Once the files are uncompressed, edit the uncompressed file `CMakeLists.txt` and add `add_subdirectory(CUDA)`. After that, execute `cmake .`, change the directory to `CUDA` and execute `make matrix_mult_cuda`.

To compile the OpenMP files, change the directory to `OMP` and `execute gcc -fopenmp -O4 -o matrix_mult_omp matrix_mult_omp.c -lm`.
Once all the files are compiled, execute `qsub submission_file.sub`, where `submission_file.sub` is any of the submission files provided. These are the regular ones, to perform the tests presented in the report, and the correctness ones, used to print the matrices for a small correctness test when the correctness section is uncommented on the source code files.

## Acknowledgements

The skeleton of the OMP source code was developed by Salvatore Filippone.
The skeleton of the CUDA source code was developed by Michał Czapiński in 2014.

<hr>

This code is my submission for Cranfield University's Small Scale Parallel Programming module assignment for the MSc in Computational and Software Techniques in Engineering - Computational Intelligence for Data Analytics.