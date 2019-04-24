//#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>
#endif
// #include <string>
#include <string.h>
extern "C" {
#include "mm.h"
}

// template <int BLOCK_SIZE> __global__
__global__
void mm_kernel_8(int block_size, float *C, float *A, float *B, int wA, int wB) {

  const int blk_size = block_size;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * blk_size * by;
  int aEnd = aBegin +wA - 1;

  int aStep = 8;

  int bBegin = 8 * bx;
  int bStep = 8 * wB;

  float Csub = 0;

  // constant value expected
  int a,b;
  for (a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep) {
    __shared__ float As[8][8];
    __shared__ float Bs[8][8];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
      Csub += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  int c = wB * 8 * by + 8 * bx;
  C[c +wB * ty + tx] = Csub;
}

__global__
void mm_kernel_16(int block_size, float *C, float *A, float *B, int wA, int wB) {

  const int blk_size = block_size;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * blk_size * by;
  int aEnd = aBegin +wA - 1;

  int aStep = 16;

  int bBegin = 16 * bx;
  int bStep = 16 * wB;

  float Csub = 0;

  // constant value expected
  int a,b;
  for (a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k)
    {
      Csub += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  int c = wB * 16 * by + 16 * bx;
  C[c +wB * ty + tx] = Csub;
}

__global__
void mm_kernel_32(int block_size, float *C, float *A, float *B, int wA, int wB) {

  const int blk_size = block_size;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * blk_size * by;
  int aEnd = aBegin +wA - 1;

  int aStep = 32;

  int bBegin = 32 * bx;
  int bStep = 32 * wB;

  float Csub = 0;

  // constant value expected
  int a,b;
  for (a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();
#pragma unroll
    for (int k = 0; k < 32; ++k)
    {
      Csub += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  int c = wB * 32 * by + 32 * bx;
  C[c +wB * ty + tx] = Csub;
}

extern "C"
void mm_gpu(int wA, int hA, int wB, int hB, int block_size, float * host_A, float * host_B, float * host_C)
{
  dim3 dimsA(wA*block_size, hA*block_size, 1);
  dim3 dimsB(wB*block_size, hB*block_size, 1);

  unsigned int size_A = dimsA.x* dimsA.y; // get size of A matrix
  unsigned int size_B = dimsB.x*dimsB.y; // get size of B matrix
  float *h_A = host_A;
  float *h_B = host_B;
  float *h_C = host_C;

  // allocate device memory
  float *d_A, *d_B, *d_C;

  dim3 dimsC(dimsB.x, dimsA.y, 1);
  // dim3 dimsC(dimsA.y, dimsB.x, 1); // height A, width B
  unsigned int mem_size_A = size_A * sizeof(float);
  unsigned int mem_size_B = size_B * sizeof(float);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

  cudaError_t error;
  error = cudaMalloc((void **) &d_A, mem_size_A);
  if (error != cudaSuccess)
  {
    printf("cudaMalloc d_A failed... \n");
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_B, mem_size_B);
  if (error != cudaSuccess)
  {
    printf("cudaMalloc d_B failed... \n");
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **) &d_C, mem_size_C);
  if (error != cudaSuccess)
  {
    printf("cudaMalloc d_C failed... \n");
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_A,h_A) failed...\n");
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (d_B, h_B) failed... \n");
    exit(EXIT_FAILURE);
  }

  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
  // dim3 grid(dimsA.y / threads.y, dimsB.x / threads.x);

  if(block_size == 8)
  {
    mm_kernel_8<<<grid, threads >>>(block_size, d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  else if(block_size == 16)
  {
    mm_kernel_16<<<grid, threads >>>(block_size, d_C, d_A, d_B, dimsA.x, dimsB.x);
  }
  else
  {
    mm_kernel_32<<<grid, threads >>>(block_size, d_C, d_A, d_B, dimsA.x, dimsB.x);
  }

  cudaDeviceSynchronize();

  error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
  {
    printf("cudaMemcpy (h_C, d_C) failed ... \n");
    exit(EXIT_FAILURE);
  }

  // TODO: Check Correctness of computed values?
  #ifdef PRINT_RES
  int i, j;
  for(i = 0; i < hA * block_size; i++)
    for(j = 0; j < wB * block_size; j++)
      printf("Result[%d][%d]: %lf\n", i, j, h_C[i * wB * block_size + j]);
  #endif

  cudaFree(d_A);
  //cudaFree(d_B);
  //cudaFree(d_C);
}

#ifdef USE_CUBLAS
extern "C"
void mm_gpu_cublas(int wA, int hA, int wB, int hB, int block_size, float *host_A, float
    *host_B, float *host_C)
{
    PRINTF("Running with CUBLAS \n");
    unsigned int size_A = hA * wA * block_size * block_size;
    unsigned int mem_size_A = size_A * sizeof(float);
    unsigned int size_B = hB * wB * block_size * block_size;
    unsigned int mem_size_B = size_B * sizeof(float);
    unsigned int size_C = hA * wB * block_size * block_size;
    unsigned int mem_size_C = size_C * sizeof(float);

    float *d_A, *d_B, *d_C;

    // checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    // checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    // checkCudaErrors(cudaMemcpy(d_A, host_A, mem_size_A, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_B, host_B, mem_size_B, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMemcpy(d_A, host_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, mem_size_C);
    dim3 threads(block_size, block_size);
    dim3 grid(wB * block_size/threads.x, hA * block_size / threads.y);
    //------
    // CUBLAS version 2
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;

    // checkCudaErrors(cublasCreate(&handle));
    // checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, wB * block_size
    //     , hA * block_size, wA * block_size, &alpha, d_B, wB * block_size, d_A
    //     , wA * block_size, &beta, d_C, wB * block_size));

    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, wB * block_size
        , hA * block_size, wA * block_size, &alpha, d_B, wB * block_size, d_A
        , wA * block_size, &beta, d_C, wB * block_size);

    cudaDeviceSynchronize();
    // checkCudaErrors(cudaMemcpy(host_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cublasDestroy(handle));
    cudaMemcpy(host_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    //------
    // check CUBLAS result?
    // TODO: Check Correctness of computed values?
#ifdef PRINT_RES
    int i, j;
    for(i = 0; i < hA * block_size; i++)
      for(j = 0; j < wB * block_size; j++)
        printf("Result[%d][%d]: %lf\n", i, j, host_C[i * wB * block_size + j]);
#endif 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
#endif

extern "C"
void mm_calculate_gpu(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
  int wA = paramv[0];
  int hA = paramv[1];
  int wB = paramv[2];
  int hB = paramv[3];
  int block_size = paramv[4];
  float * h_A = (float *)depv[0].ptr;
  float * h_B = (float *)depv[1].ptr;
  float * h_C = (float *)depv[2].ptr;

  // check size of dimensions of Matrices A and B.
#ifdef USE_CUBLAS
  mm_gpu_cublas(wA, hA, wB, hB, block_size, h_A, h_B, h_C);
#else
  mm_gpu(wA, hA, wB, hB, block_size, h_A, h_B, h_C);
#endif
  artsSignalEdt(paramv[5], 0, NULL_GUID);
}
