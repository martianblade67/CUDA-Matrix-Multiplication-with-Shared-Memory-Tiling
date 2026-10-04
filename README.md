# CUDA Matrix Multiplication

A CUDA project implementing square matrix multiplication using both a basic CUDA kernel and a tiled CUDA kernel with shared memory. The project compares the performance of CPU multiplication, basic GPU multiplication, and tiled GPU multiplication.

## Features

* Square matrix generation with random integer values
* CPU matrix multiplication
* Basic CUDA matrix multiplication
* Tiled CUDA matrix multiplication using shared memory
* Configurable matrix size
* CUDA event-based GPU timing
* Multiple runs with a warm-up run
* Average execution time calculation
* Comparison between CPU and GPU implementations

## Implementations

### 1. CPU Matrix Multiplication

The CPU implementation performs the standard matrix multiplication algorithm:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

Each element of the output matrix is calculated using three nested loops.

### 2. Basic CUDA Matrix Multiplication

The basic CUDA implementation assigns one GPU thread to calculate one element of the output matrix.

Each thread calculates:

```text
C[row][col]
```

by iterating through the corresponding row of `A` and column of `B`.

### 3. Tiled CUDA Matrix Multiplication

The tiled implementation uses CUDA shared memory to reduce repeated accesses to global memory.

The matrices are divided into `16 × 16` tiles. Each thread loads an element into shared memory, synchronizes with the other threads in the block, and then uses the shared-memory tiles to calculate the output.

This improves memory reuse compared with the basic implementation.

## Timing

GPU execution time is measured using CUDA events:

```cpp
cudaEventRecord(start);
```

and

```cpp
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

The first execution is used as a warm-up run. The following five runs are timed and averaged to reduce the effect of initialization and first-run overhead.

## Requirements

* NVIDIA GPU with CUDA support
* NVIDIA CUDA Toolkit
* C++ compiler compatible with CUDA
* Windows/Linux environment capable of compiling CUDA programs

## Compilation

Compile the CUDA source using `nvcc`.

For example:

```bash
nvcc matrix_multiplication.cu -o matrix_multiplication.exe
```

Run it with:

```bash
./matrix_multiplication.exe
```

On Windows Command Prompt:

```cmd
matrix_multiplication.exe
```

## Usage

When the program starts, enter the size of the square matrices.

For example:

```text
Enter matrix size: 1000
```

The program generates the matrices and runs the CPU, basic CUDA, and tiled CUDA implementations.

The execution times and calculated results are then displayed.

## Project Structure

```text
CUDA-Matrix-Multiplication/
│
├── matrix_multiplication.cu
└── README.md
```

## Concepts Demonstrated

This project demonstrates several CUDA and parallel computing concepts:

* CUDA kernels
* Threads and thread indexing
* Blocks and grids
* Global memory
* Shared memory
* Thread synchronization
* Tiled matrix multiplication
* CUDA event timing
* GPU memory allocation and transfers
* CPU vs GPU performance comparison

## Purpose

The purpose of this project is to study how matrix multiplication can be parallelized on a GPU and to compare a straightforward CUDA implementation with an optimized tiled implementation that makes use of shared memory.

The project also demonstrates the performance difference between CPU execution and GPU execution for computationally intensive matrix operations.
