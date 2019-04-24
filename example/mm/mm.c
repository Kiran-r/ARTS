/*Matrix Multiply Example*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>

#include "mm.h"
#include <string.h>

uint64_t start = 0;

int wA = 1;          // Default Width of A
int hA = 1;          // Default Height of A
int wB = 2;          // Default Width of B
int hB = 1;          // Default Height of B
int block_size = 32; // Default Block Size;

artsGuid_t dbGuidMatA = NULL_GUID;
artsGuid_t dbGuidMatB = NULL_GUID;
artsGuid_t dbGuidMatC = NULL_GUID;

void initializeMatrix(float seed, int n, int m, float * data) 
{
    for (unsigned int i = 0; i < m; i++) 
    {
        for (unsigned int j = 0; j < n; j++) 
            data[i * m + j] = (float) (seed * j);
    }

//    for (unsigned int i = 0; i < m; i++) 
//    {
//        for (unsigned int j = 0; j < n; j++) 
//        {
//            printf("Data[%d][%d]: %lf\n",i, j, data[i * m + j]);
//        }
//    }
}

void mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    int wA = (int) paramv[0];
    int hA = (int) paramv[1];
    int wB = (int) paramv[2];
    int hB = (int) paramv[3];
    int block_size = (int) paramv[4];
    int m = hA * block_size;
    int n = wB * block_size;
    int o = hB * block_size; // wA * block_size;

    float * matA = (float *) (depv[0].ptr);
    float * matB = (float *) (depv[1].ptr);
    float * matC = (float *) (depv[2].ptr);
    int i, j, k;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < o; ++k)
                matC[i * n + j] += matA[i * o + k] * matB[k * n + j];
    #ifdef PRINT_RES
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            printf("Result[%d][%d]: %lf\n", i, j, matC[i * n + j]);
    #endif
    artsSignalEdt(paramv[5], 0, NULL_GUID);
}

void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t time = artsGetTimeStamp() - start;
    PRINTF("MM completed! Time: %lu\n", time);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    if(argc == 6)
    {
        wA = atoi(argv[1]);
        hA = atoi(argv[2]);
        wB = atoi(argv[3]);
        hB = atoi(argv[4]);
        block_size = atoi(argv[5]);
    }
    
    dbGuidMatA = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    dbGuidMatB = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    dbGuidMatC = artsReserveGuidRoute(ARTS_DB_PIN, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {
    if (!nodeId && !workerId) 
    {
        float * ptrA = artsDbCreateWithGuid(dbGuidMatA, wA * hA * block_size * block_size * sizeof (float));
        float * ptrB = artsDbCreateWithGuid(dbGuidMatB, wB * hB * block_size * block_size * sizeof (float));
        float * ptrC = artsDbCreateWithGuid(dbGuidMatC, hA * wB * block_size * block_size * sizeof (float));
        initializeMatrix(.5, wA * block_size, hA * block_size, ptrA);
        initializeMatrix( 1, wB * block_size, hB * block_size, ptrB);
        initializeMatrix( 0, wB * block_size, hA * block_size, ptrC);
        
        start = artsGetTimeStamp();
        artsGuid_t doneGuid = artsEdtCreate(shutDown, 0, 0, NULL, 1);
        uint64_t args[6] = {
            (uint64_t) wA, 
            (uint64_t) hA, 
            (uint64_t) wB, 
            (uint64_t) hB, 
            (uint64_t) block_size,
            doneGuid
        };
        uint64_t argM[5] = {(uint64_t) wA, (uint64_t) hA, (uint64_t) wB, (uint64_t) hB, (uint64_t) block_size};
        #ifdef USE_GPU
            artsGuid_t mmCalcGuid = artsEdtCreate(mm_calculate_gpu, 0, 6, args, 3);
            PRINTF("mmCalcGuid guid(gpu): %lu, func_ptr: %p\n", mmCalcGuid, (void *)mm_calculate_gpu);
            
        #else
            artsGuid_t mmCalcGuid = artsEdtCreate(mm_calculate, 0, 6, args, 3);
            PRINTF("mmCalcGuid guid: %lu, func_ptr: %p\n", mmCalcGuid, (void *)mm_calculate);
        #endif
            
        artsSignalEdt(mmCalcGuid, 0, dbGuidMatA);
        artsSignalEdt(mmCalcGuid, 1, dbGuidMatB);
        artsSignalEdt(mmCalcGuid, 2, dbGuidMatC);
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
