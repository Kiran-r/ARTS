#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H
#include "arts.h"

//#define PRINT_RES 1

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

void mm(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv []);
void mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
#ifdef USE_GPU
void mm_calculate_gpu(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void mm_gpu(int wA, int hA, int wB, int hB, int block_size, float *host_A, float *host_B, float *host_C);
#ifdef USE_CUBLAS
void mm_gpu_cublas(int wA, int hA, int wB, int hB, int block_size, float *host_A, float *host_B, float *host_C);
#endif
#endif
#endif
