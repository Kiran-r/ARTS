/*Matrix Multiply Example*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
// #include "hiveRT.h"

#include "mm.h"
#include <string.h>

//artsGuid_t initializeMatrix(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
void initializeMatrix(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {

  float * data = (float *)depv[0].ptr;
  float *seed = (float *)paramv[0];
  // printf("Seed: %lf.\n", *seed);
  int n = paramv[1]; // width
  int m = paramv[2]; // height
  int i = 0, j = 0;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++){
      data[i * m + j] = (float)((*seed) * j);
    }
  }

  /*
    for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("Data[%d][%d]: %lf\n",i, j, data[i * m + j]);
    }
  }
  */

  // printf("Initialize Matrix Called!\n");
  artsGuid_t funcCompleteGuid = NULL_GUID;
  // artsSignalEdt(funcCompleteGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
  artsSignalEdtValue(funcCompleteGuid, 0, NULL_GUID);
}

// artsGuid_t mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
void mm_calculate(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {

  int wA = (int)paramv[0];
  int hA = (int)paramv[1];
  int wB = (int)paramv[2];
  int hB = (int)paramv[3];
  int block_size = (int)paramv[4];
  int m = hA * block_size;
  int n = wB * block_size;
  int o = hB * block_size; // wA * block_size;

  float * matA = (float *)(depv[0].ptr);
  float * matB = (float *)(depv[1].ptr);
  float * matC = (float *)(depv[2].ptr);
  int i,j,k;
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      for (k = 0; k < o; ++k)
        matC[i * n + j] += matA[i * o + k] * matB[k * n + j];

  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      printf("Result[%d][%d]: %lf\n", i, j, matC[i * n + j]);
}

// artsGuid_t mm(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
void mm(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {

  int block_size = 0;
  int wA, hA, wB, hB;
  artsGuid_t shutDownGuid = paramv[0];
  if(paramc > 1) {
    wA = paramv[1];
    hA = paramv[2];
    wB = paramv[3];
    hB = paramv[4];
    block_size = paramv[5];
  }
  else {
    wA = 1; // Default Width of A
    hA = 1; // Default Height of A
    wB = 2; // Default Width of B
    hB = 1; // Default Height of B
    block_size = 32; // Default Block Size;
  }
  float * seedA = (float *) malloc(sizeof(float));
  *seedA = 0.5;
  float * seedB = (float *) malloc(sizeof (float));
  *seedB = 1;
  float * seedC = (float *) malloc(sizeof(float));
  *seedC = 0;
  float * ptrA;
  float *ptrB;
  artsGuid_t dbGuidMatA = artsDbCreate((void **)&ptrA
    , wA * hA * block_size * block_size * sizeof(float), ARTS_DB_READ);
  artsGuid_t dbGuidMatB = artsDbCreate((void **)&ptrB
    , wB * hB * block_size * block_size * sizeof(float), ARTS_DB_READ);
  float *ptrC;
  artsGuid_t dbGuidMatC = artsDbCreate((void **)&ptrC
    , hA * wB * block_size * block_size * sizeof(float), ARTS_DB_WRITE);

  uint64_t argsA[3] = {(uint64_t)seedA, (uint64_t)(wA * block_size), (uint64_t)(hA * block_size)};
  // Create Function Ptr
  artsGuid_t initA = artsEdtCreate(initializeMatrix, 0, 3, argsA, 1);
  // PRINTF("InitA guid: %lu, func_ptr: %p\n", initA, (void *)initializeMatrix);
  // Signal Function

  uint64_t argsB[3] = {(uint64_t)seedB, (uint64_t)(wB * block_size), (uint64_t)(hB * block_size)};
  artsGuid_t initB = artsEdtCreate(initializeMatrix, 0, 3, argsB, 1);
  // PRINTF("InitB guid: %lu, func_ptr: %p\n", initB, (void *)initializeMatrix);

  // u64 argsC[3] = {(u64) seedC, DB_SIZE, DB_SIZE};
  uint64_t argsC[3] = {(uint64_t)seedC, (uint64_t)(wB * block_size), (uint64_t)(hA * block_size)};
  artsGuid_t initC = artsEdtCreate(initializeMatrix, 0, 3, argsC, 1);
  // PRINTF("InitC guid: %lu, func_ptr: %p\n", initC, (void *)initializeMatrix);

  uint64_t argM[5] = {(uint64_t)wA, (uint64_t)hA, (uint64_t)wB, (uint64_t)hB, (uint64_t)block_size};
#ifdef USE_GPU
  artsGuid_t mmCalcGuid = artsEdtCreate(mm_calculate_gpu, 0,5,argM,3);
  // PRINTF("mmCalcGuid guid(gpu): %lu, func_ptr: %p\n", mmCalcGuid, (void *)mm_calculate_gpu);
#else
  artsGuid_t mmCalcGuid = artsEdtCreate(mm_calculate, 0,5,argM,3);
  // PRINTF("MMCALCGUID guid: %lu, func_ptr: %p\n", mmCalcGuid, (void *)mm_calculate);
#endif
  artsSignalEdt(mmCalcGuid, 2, dbGuidMatC);
  artsSignalEdt(mmCalcGuid, 1, dbGuidMatB);
  artsSignalEdt(mmCalcGuid, 0, dbGuidMatA);

  artsSignalEdt(initC, 0, dbGuidMatC);
  artsSignalEdt(initB, 0, dbGuidMatB);
  artsSignalEdt(initA, 0, dbGuidMatA);

  artsSignalEdt(shutDownGuid, 0, NULL_GUID);

}

// artsGuid_t shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
  PRINTF("MM completed!\n");
  artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc,
  char** argv)
{
  if(!nodeId && !workerId)
  {
    // argv[1] - wA
    // argv[2] - hA
    // argv[3] - wB
    // argv[4] - hB
    // argv[5] - block_size: 8, 16 and 32 supported for now.
    artsGuid_t doneGuid = artsEdtCreate(shutDown, 0, 0, NULL, 0);
    if(argc > 5) {
      if(atoi(argv[1]) != atoi(argv[4])) {
        PRINTF("Matrix Outer dimension don't match!\n");
        exit(EXIT_FAILURE);
      }
      PRINTF("Taking User Input.\n");
      uint64_t args[6] = {doneGuid
                   ,(uint64_t)(atoi(argv[1]))
                   , (uint64_t)(atoi(argv[2]))
                   , (uint64_t)(atoi(argv[3]))
                   , (uint64_t)(atoi(argv[4]))
                   , (uint64_t)(atoi(argv[5]))
                   };
      artsEdtCreate(mm, 0,6, args,0);
    }
    else {
      PRINTF("Taking Default Values.\n");
      uint64_t args[1] = {doneGuid};
      artsEdtCreate(mm, 0,1, args,0);
    }
  }
}

int main(int argc, char** argv)
{
  artsRT(argc, argv);
  return 0;
}
