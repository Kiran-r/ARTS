/*Matrix Multiply Example*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#include "mm.h"
#include <string.h>

uint64_t time = 0;

int wA = 1;          // Default Width of A
int hA = 1;          // Default Height of A
int wB = 2;          // Default Width of B
int hB = 1;          // Default Height of B
int block_size = 32; // Default Block Size;

void initializeMatrix(float seed, int n, int m, float * data) 
{
    for (unsigned int i = 0; i < m; i++) 
    {
        for (unsigned int j = 0; j < n; j++) 
            data[i * m + j] = (float) (seed * j);
    }
}

int main(int argc, char** argv) {
    if(argc == 6)
    {
        wA = atoi(argv[1]);
        hA = atoi(argv[2]);
        wB = atoi(argv[3]);
        hB = atoi(argv[4]);
        block_size = atoi(argv[5]);
    }
    
    float * ptrA = (float*) malloc(wA * hA * block_size * block_size * sizeof (float));
    float * ptrB = (float*) malloc(wB * hB * block_size * block_size * sizeof (float));
    float * ptrC = (float*) malloc(hA * wB * block_size * block_size * sizeof (float));
    initializeMatrix(.5, wA * block_size, hA * block_size, ptrA);
    initializeMatrix( 1, wB * block_size, hB * block_size, ptrB);
    initializeMatrix( 0, wB * block_size, hA * block_size, ptrC);
    
    time = artsGetTimeStamp();
    
    #ifdef USE_CUBLAS
        mm_gpu_cublas(wA, hA, wB, hB, block_size, ptrA, ptrB, ptrC);
    #elif USE_GPU 
        mm_gpu(wA, hA, wB, hB, block_size, ptrA, ptrB, ptrC);
    #else
        int m = hA * block_size;
        int n = wB * block_size;
        int o = hB * block_size; // wA * block_size;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; ++k)
                    ptrC[i * n + j] += ptrA[i * o + k] * ptrB[k * n + j];
    #endif
    
    time = artsGetTimeStamp() - time;
    PRINTF("MM completed! Time: %lu\n", time);
    return 0;
}
