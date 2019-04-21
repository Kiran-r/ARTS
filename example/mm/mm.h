#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H
/*
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
*/
#include "arts.h"

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

// artsGuid_t initializeMatrix(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void initializeMatrix(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
// artsGuid_t mm(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv []);
void mm(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv []);
// artsGuid_t mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
#ifdef USE_GPU
// artsGuid_t mm_calculate_gpu(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void mm_calculate_gpu(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
// int mm_gpu(int block_size, dim3 *dimsA, dim3 *dimsB, float *host_A, float *host_B, float *host_C);
// int mm_gpu(int block_size, float *host_A, float *host_B, float *host_C);
void mm_gpu(int wA, int hA, int wB, int hB, int block_size, float *host_A, float *host_B, float *host_C);
#ifdef USE_CUBLAS
void mm_gpu_cublas(int wA, int hA, int wB, int hB, int block_size, float *host_A, float *host_B, float *host_C);
#endif
// int mm_calculate_gpu_helper(int block_size, float * h_A, float * h_B, float *h_C);
#endif

#endif
