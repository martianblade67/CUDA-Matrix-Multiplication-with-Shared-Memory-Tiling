# CUDA-Matrix-Multiplication-with-Shared-Memory-Tiling
A CUDA implementation of square matrix multiplication using **shared-memory tiling** to improve memory access efficiency and parallel computation on the GPU.

## Overview

This project implements matrix multiplication:

C = A × B

using CUDA. Each CUDA thread computes one element of the resulting matrix, while portions of the input matrices are loaded into **shared memory** in 16×16 tiles.

The implementation also measures execution time across multiple runs, using the first run as a warm-up and averaging the remaining runs.

## Features

- Parallel matrix multiplication using CUDA
- 16×16 CUDA thread blocks
- Shared-memory tiling
- Boundary handling for matrix sizes that are not multiples of the tile size
- Random matrix generation
- Warm-up execution
- Average execution-time measurement across multiple runs
- Result verification by copying the output matrix back to the CPU
