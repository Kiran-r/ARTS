#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <artsGpuStream.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#include <string.h>
extern "C" {
#include "vecAdd.h"
}

#define TILE_WIDTH 16


__global__ void vector_add(float *A, float *B, float *C, int n) {
    int id = blockIdx.x * blockDim.x+threadIdx.x; 
	if (id < n)
		C[id] = A[id] + B[id];
}

extern "C"
void vecAddStream(int num_elems, int block_size, float* host_A, float* host_B, float* host_C)
{
    int num_streams = 32; 
    cudaStream_t streams[num_streams];
    int i;
    int bytes = vec_length * sizeof(float);
    int elems_per_stream = vec_lengty/num_streams;
    // host pinned array
    float *hP_A, *hP_B, *hP_C;
    CHECKCORRECT(cudaHostAlloc(&hP_A, bytes, cudaHostAllocDefault));
    memcpy(hP_A, host_A, bytes);
    CHECKCORRECT(cudaHostAlloc(&hP_B, bytes, cudaHostAllocDefault));
    memcpy(hP_B, host_B, bytes);
    CHECKCORRECT(cudaHostAlloc(&hP_C, bytes, cudaHostAllocDefault));
    // memcpy(hP_C, host_C, bytes);

	// Device input vectors
    float *d_A;
    float *d_B;
    //Device output vector
    float *d_C;

	int blockSize = 1024;
	int gridSize = (int)ceil((float)num_elems/blockSize);
    
    for (i = 0; i < num_streams; i++) {
        int offset = i * elems_per_stream;
        cudaMemcpyAsync(&d_A[offset], &hP_A[offset]
        , elems_per_stream * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_B[offset], &hP_B[offset]
        , elems_per_stream * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        vector_add<<<gridSize, blockSize, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, elems_per_stream);
        cudaMemcpyAsnyc(&hP_C[offset], &d_C[offset], elems_per_stream * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (i = 0; i < num_streams; i++)
        cudaStreamSynchronize(streams[i]);

    memcpy(host_C, hP_C, bytes);

    cudaFree(hP_A);
    cudaFree(hP_B);
    cudaFree(hP_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
extern "C"
void vecAddStreamArts(int num_elems, int block_size, float* host_A, float* host_B, float* host_C)
{

// Use Stream by default
extern "C"
void vecAddGPU(uint32_t paramc, uint64_t *paramv, uint32_t depc, artsEdtDep_t depv[])
{
    int stream_count = 32; 
    int length = paramv[0];
    int block_size = paramv[1];
    int stream_count = paramv[2];
    int stream_count = paramv[2];
    float * h_A = (float *)depv[0].ptr; 
    float * h_B = (float *)depv[1].ptr; 
    float * h_C = (float *)depv[2].ptr;
    vecAddStreamArts(length, block_size, h_A, h_B, h_C);

    artsSignalEdt(paramv[3], 0, NULL_GUID);
}
