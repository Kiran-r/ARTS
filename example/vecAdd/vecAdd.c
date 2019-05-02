
#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsGpuRuntime.h"

uint64_t start = 0;
int num_elems = 1000000;
int block_size = 1024;

void initializeArray(float seed, int n, float * data) 
{
    for (unsigned int i = 0; i < n; i++) 
        data[i] = (float) (seed * i);
}

void vecAdd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    int numElems = (int) paramv[0];
    int blockSize = (int) paramv[1];

    float * arrA = (float*) (depv[0].ptr)
    float * arrB = (float*) (depv[1].ptr)
    float * arrC = (float*) (depv[2].ptr)
    int i; 
    for (i=0; i<numElems; i++)
        arrC[i] = arrA[i] + arrB[i];

    artsSignalEdt(paramv[2], 0, NULL_GUID);
}

void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t time = artsGetTimeStamp() - start;
    PRINTF("VecAdd completed! Time: %lu\n", time);
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
    if(argc == 3)
    {
        num_elems = atoi(argv[1]);
        block_size = atoi(argv[2]);
    }
    
    //Create two DB of type ARTS_DB_GPU
    float * ptrA = NULL, *ptrB = NULL, *ptrC = NULL;
    artsGuid_t dbGuidArrA = artsDbCreate((void**)&ptrA, sizeof(float)*num_elems, ARTS_DB_GPU);
    artsGuid_t dbGuidArrB = artsDbCreate((void**)&ptrB, sizeof(float)*num_elems, ARTS_DB_GPU);
    artsGuid_t dbGuidArrC = artsDbCreate((void**)&ptrC, sizeof(float)*num_elems, ARTS_DB_GPU);

}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        float * ptrA = artsDbCreateWithGuid(dbGuidArrA, num_elems * sizeof (float));
        float * ptrB = artsDbCreateWithGuid(dbGuidArrB, num_elems * sizeof (float));
        float * ptrC = artsDbCreateWithGuid(dbGuidMatC, num_elems * sizeof (float));

        initializeArray(0.5, num_elems, ptrA);
        initializeArray(1, num_elems, ptrB);
        initializeArray(0, num_elems, ptrC);

        start = artsGetTimeStamp();
        artsGuid_t doneGuid = artsEdtCreate(shutDown, 0, 0, NULL, 1);
        uint64_t args[3] = {
            (uint64_t) num_elems, 
            (uint64_t) block_size,
            doneGuid
        };
        
        // uint64_t argA[2] = {(uint64_t) num_elems, (uint64_t) block_size};
        #ifdef USE_GPU
            artsGuid_t vecAddGuid = artsEdtCreate(vecAddGPU, 0, 3, args, 3);
            PRINTF("vecAddGuid guid(gpu): %lu, func_ptr: %p\n", vecAddGuid, (void *)vecAddGPU);
        #else
            artsGuid_t vecAddGuid = artsEdtCreate(vecAdd, 0, 3, args, 3);
            PRINTF("vecAddGuid guid: %lu, func_ptr: %p\n", vecAddGuid, (void *)vecAdd);
        #endif

        artsSignalEdt(vecAddGuid, 0, dbGuidArrA);
        artsSignalEdt(vecAddGuid, 1, dbGuidArrB);
        artsSignalEdt(vecAddGuid, 2, dbGuidArrC);
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
