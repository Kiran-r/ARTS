/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,* 
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#ifndef ARTSGPUSTREAM_H
#define ARTSGPUSTREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>    
#include "artsRT.h"
#include "artsAtomics.h"
#include "artsArrayList.h"

#define CHECKCORRECT(x) {                                   \
  cudaError_t err;                                          \
  if( (err = (x)) != cudaSuccess )                          \
    PRINTF("FAILED %s: %s\n", #x, cudaGetErrorString(err)); \
}
    
typedef struct
{
    volatile unsigned int * scheduleCounter;
    volatile unsigned int * deleteLock;
    volatile unsigned int * newEdtLock;
    artsArrayList * deleteQueue;
    artsArrayList * deleteHostQueue;
    artsArrayList * newEdts;
    void * devDB;
    void * devClosure;
    struct artsEdt * edt;
} artsGpuCleanUp_t;

typedef struct 
{
    int device;
    volatile size_t memUtil;            // Memory Utilization in device
    volatile unsigned int scheduled;    // Count(Edts) on device
    cudaStream_t stream;
    volatile unsigned int deleteLock;
    artsArrayList * deleteQueue;
    artsArrayList * deleteHostQueue;
    uint64_t devSize;
    uint64_t hostSize;
} artsGpu_t;

void artsInitGpus(unsigned int entries, unsigned int tableSize, int numGpus);
void artsCleanupGpus();
void artsScheduleToGpuInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr, artsGpu_t * artsGpu);
void artsScheduleToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr, artsGpu_t * artsGpu);
struct artsRouteItem * artsGpuRouteTableSearchForKey(artsGuid_t key, int gpuId);

void artsGpuSynchronize(artsGpu_t * artsGpu);
void artsGpuStreamBusy(artsGpu_t* artsGpu);
artsGpu_t * artsGpuScheduled(unsigned id);

void artsStoreNewEdts(void * edt);
void artsHandleNewEdts();
void artsFreeGpuMemory(artsGpu_t * artsGpu);
void artsGpuFree(void * data, unsigned int gpu);
void * artsGpuHostToDeviceDbs (uint32_t depc, uint64_t * paramv, artsEdtDep_t * depv, int gpuId, artsGuid_t * edtGuid, artsGpu_t * artsGpu, void ** devParamv);
void artsScheduleKernelToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * gpuParamv, uint32_t depc, artsEdtDep_t * gpuDepv, dim3 grid, dim3 block, artsGpu_t * artsGpu);
void artsGpuDeviceToHostDbs (uint32_t paramc,  uint32_t depc, artsEdtDep_t * depv, artsEdtDep_t * devDepv, artsGpu_t * artsGpu, void * writeDbs);
artsGpu_t * artsFindGpu(void * data, unsigned int seed);

extern int artsNumGpus;
extern artsGpu_t * artsGpus;


#ifdef __cplusplus
}
#endif

#endif /* ARTSGPUSTREAM_H */

