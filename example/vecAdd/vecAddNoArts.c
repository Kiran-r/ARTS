/* VectorAdd Example*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#include "vecAdd.h"
#include <string.h>

uint64_t time = 0;

int num_elems = 1000000;
int block_size = 1024;

void initializeArray(float seed, int n, float * data) 
{
    for (unsigned int i = 0; i < n; i++) 
        data[i] = (float) (seed * i);
}


int main(int argc, char** argv) {
    if(argc == 3)
    {
        num_elems = atoi(argv[1]);
        block_size = atoi(argv[2]);
    }
    
    float * ptrA = (float*) malloc(num_elems * sizeof (float));
    float * ptrB = (float*) malloc(num_elems * sizeof (float));
    float * ptrC = (float*) malloc(num_elems * sizeof (float));
    initializeArray(.5, num_elems, ptrA);
    initializeArray(1, num_elems, ptrB);
    initializeArray(0, num_elems, ptrC);
    
    time = artsGetTimeStamp();
    
    #ifdef USE_GPU 
        vecAddStream(num_elems, block_size, ptrA, ptrB, ptrC);
    #else
        for (int i = 0; i < num_elems; i++)
            ptrC[i] = ptrA[i] * ptrB[i];
    #endif
    
    time = artsGetTimeStamp() - time;
    PRINTF("VectorAdd completed! Time: %lu\n", time);
    return 0;
}
