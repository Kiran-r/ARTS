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

//Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
//and https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
//Once this *class* works we will put a stream(s) in create a thread local
//stream.  Then we will push stuff!
#include "artsDbFunctions.h"
#include "artsGpuStream.h"
#include "artsGpuRuntime.h"
#include "artsGlobals.h"
#include "artsDeque.h"
#include "artsGpuRouteTable.h"
#include "artsDebug.h"

#define DPRINTF( ... )
// #define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

volatile unsigned int hits = 0;
volatile unsigned int misses = 0;
volatile unsigned int falseMisses = 0;
volatile unsigned int freeBytes = 0;

int artsNumGpus = 0;
artsGpu_t * artsGpus;

__thread volatile unsigned int * newEdtLock = 0;
__thread artsArrayList * newEdts = NULL;

void artsNodeInitGpus(unsigned int entries, unsigned int tableSize, int numGpus)
{
    int numAvailGpus = 0;
    CHECKCORRECT(cudaGetDeviceCount(&numAvailGpus));
    if(numAvailGpus < numGpus)
    {
        PRINTF("Requested %d gpus but only %d available\n", numAvailGpus, numGpus);
        artsNumGpus = numAvailGpus;
    }
    else
        artsNumGpus = numGpus;

    DPRINTF("NUM DEV: %d\n", artsNumGpus);
    artsGpus = (artsGpu_t *) artsCalloc(sizeof(artsGpu_t) * artsNumGpus);

    int savedDevice;
    cudaGetDevice(&savedDevice);

    // Initialize artsGpu with 1 stream/GPU
    for (int i=0; i<artsNumGpus; ++i)
    {
        artsGpus[i].device = i;
        DPRINTF("Setting %d\n", i);
        CHECKCORRECT(cudaSetDevice(i));
        CHECKCORRECT(cudaStreamCreate(&artsGpus[i].stream)); // Make it scalable
        artsNodeInfo.gpuRouteTable[i] = artsGpuNewRouteTable(entries, tableSize);
    }
    CHECKCORRECT(cudaSetDevice(savedDevice));
}

void artsWorkerInitGpus()
{
    newEdtLock = (unsigned int*) artsCalloc(sizeof(unsigned int));
    newEdts = artsNewArrayList(sizeof(void*), 32);
}

void artsStoreNewEdts(void * edt)
{
    artsLock(newEdtLock);
    artsPushToArrayList(newEdts, &edt);
    artsUnlock(newEdtLock);
}

void artsHandleNewEdts()
{
    artsLock(newEdtLock);
    uint64_t size = artsLengthArrayList(newEdts);
    if(size)
    {
        for(uint64_t i=0; i<size; i++)
        {
            struct artsEdt ** edt = (struct artsEdt**) artsGetFromArrayList(newEdts, i);
            if((*edt)->header.type == ARTS_EDT)
                artsDequePushFront(artsThreadInfo.myDeque, (*edt), 0);
            if((*edt)->header.type == ARTS_GPU_EDT)
                artsDequePushFront(artsThreadInfo.myGpuDeque, (*edt), 0);
        }
        artsResetArrayList(newEdts);
    }    
    artsUnlock(newEdtLock);
}

void artsCleanupGpus()
{
    int savedDevice;
    cudaGetDevice(&savedDevice);
    for (int i=0; i<artsNumGpus; i++)
    {
        CHECKCORRECT(cudaSetDevice(artsGpus[i].device));
        CHECKCORRECT(cudaStreamSynchronize(artsGpus[i].stream));
        CHECKCORRECT(cudaStreamDestroy(artsGpus[i].stream));
    }
    CHECKCORRECT(cudaSetDevice(savedDevice));
    PRINTF("HITS: %u MISSES: %u FALSE MISSES: %u FREED BYTES: %u\n", hits, misses, falseMisses, freeBytes);
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpuCleanUp_t * gc = (artsGpuCleanUp_t*) data;
    
    artsGpu_t * artsGpu = &artsGpus[gc->gpuId];
    artsAtomicSub(&artsGpu->scheduled, 1U);

    //Shouldn't have to touch newly ready edts regardless of streams and devices
    artsGpuEdt_t * edt      = (artsGpuEdt_t *) gc->edt;
    uint32_t       paramc   = edt->wrapperEdt.paramc;
    uint32_t       depc     = edt->wrapperEdt.depc;
    uint64_t     * paramv   = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv     = (artsEdtDep_t *)(paramv + paramc);

    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
            //True says to mark it for deletion... Change this to false to further delay delete!
            artsGpuRouteTableReturnDb(depv[i].guid, true, gc->gpuId);
            DPRINTF("Returning Db: %lu id: %d\n", depv[i].guid, gc->gpuId);
        }
    }

    //Definitely mark the dev closure to be deleted as there is no reuse!
    artsGpuRouteTableReturnDb(edt->wrapperEdt.currentEdt, true, gc->gpuId);
    newEdtLock = gc->newEdtLock;
    newEdts    = gc->newEdts;
    artsGpuHostWrapUp(gc->edt, edt->endGuid, edt->slot, edt->dataGuid);
    DPRINTF("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
}

void artsScheduleToGpuInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr, artsGpu_t * artsGpu)
{
//    For now this should push the following into the stream:
//    1. Copy data from host to device
//    2. Push kernel
//    3. Copy data from device to host
//    4. Call host callback function artsGpuHostWrapUp

    static volatile unsigned int Gpulock;

    void * devClosure  = NULL;
    void * hostClosure = NULL;

    uint64_t         * devParamv  = NULL;
    artsEdtDep_t     * devDepv    = NULL;

    artsGpuCleanUp_t * hostGCPtr = NULL;
    uint64_t         * hostParamv = NULL;
    artsEdtDep_t     * hostDepv   = NULL;

    DPRINTF("Paramc: %u Depc: %u edt: %p\n", paramc, depc, edtPtr);

    //Get size of closure
    size_t devClosureSize = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    size_t hostClosureSize = devClosureSize + sizeof(artsGpuCleanUp_t);
    DPRINTF("devClosureSize: %u hostClosureSize: %u\n", devClosureSize, hostClosureSize);

    //Allocate Closure for GPU
    if(devClosureSize)
    {
        CHECKCORRECT(cudaMalloc(&devClosure, devClosureSize));
        devParamv = (uint64_t*) devClosure;
        devDepv = (artsEdtDep_t *)(devParamv + paramc);
        DPRINTF("Allocated dev closure\n");
    }

    if(hostClosureSize)
    {
        //Allocate closure for host
        CHECKCORRECT(cudaMallocHost(&hostClosure, hostClosureSize));
        hostGCPtr = (artsGpuCleanUp_t *) hostClosure;
        hostParamv = (uint64_t*)(hostGCPtr + 1);
        hostDepv = (artsEdtDep_t *)(hostParamv + paramc);
        DPRINTF("Allocated host closure\n");

        //Fill Host closure
        hostGCPtr->gpuId = artsGpu->device;
        hostGCPtr->newEdtLock = newEdtLock;
        hostGCPtr->newEdts = newEdts;
        hostGCPtr->devClosure = devClosure;
        hostGCPtr->edt = (struct artsEdt*)edtPtr;
        for(unsigned int i=0; i<paramc; i++)
            hostParamv[i] = paramv[i];
        DPRINTF("Filled host closure\n");

        artsGuid_t edtGuid = hostGCPtr->edt->currentEdt;
        artsGpuRouteTableAddItemRace(hostGCPtr, hostClosureSize, edtGuid, artsGpu->device);
        DPRINTF("Added edtGuid: %lu size: %u to gpu: %d routing table\n", edtGuid, hostClosureSize, artsGpu->device);        
    }

    //Allocate space for DB on GPU and Move Data
    for (unsigned int i=0; i<depc; ++i)
    {
        if(depv[i].ptr)
        {
            void * dataPtr = artsGpuRouteTableLookupDb(depv[i].guid, artsGpu->device);
            if (!dataPtr)
            {
                //Actually allocate space
                struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
                size_t size = db->header.size - sizeof(struct artsDb);
                CHECKCORRECT(cudaMalloc(&dataPtr, size));
                void * newDataPtr = artsGpuRouteTableAddItemRace(dataPtr, size, depv[i].guid, artsGpu->device);
                if(newDataPtr == dataPtr) //We won, so move data
                {
                    DPRINTF("Adding %lu %u id: %d\n", depv[i].guid, size, artsGpu->device);
                    CHECKCORRECT(cudaMemcpyAsync(dataPtr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsGpu->stream));
                    artsAtomicAdd(&misses, 1U);
                }
                else //Someone beat us to creating the data... So we must free
                {
                    //Bug here...
                    cudaFree(dataPtr);
                    dataPtr = newDataPtr;
                    artsAtomicAdd(&falseMisses, 1U);
                }
            }
            else
                artsAtomicAdd(&hits, 1U);
            
            hostDepv[i].ptr = dataPtr;
        }
        else
            hostDepv[i].ptr = NULL;

        hostDepv[i].guid = depv[i].guid;
        hostDepv[i].mode = depv[i].mode;    
    }
    DPRINTF("Allocated, added, and moved dbs\n");
    
    CHECKCORRECT(cudaMemcpyAsync(devClosure, (void*)hostParamv, devClosureSize, cudaMemcpyHostToDevice, artsGpu->stream));
    DPRINTF("Filled GPU Closure\n");
    
    //Launch kernel
    fnPtr<<<grid, block, 0, artsGpu->stream>>>(paramc, devParamv, depc, devDepv);
    
    //Move data back
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr && depv[i].mode == ARTS_DB_GPU_WRITE)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, artsGpu->stream));
        }
    }

#if CUDART_VERSION >= 10000
    // CHECKCORRECT(cudaLaunchHostFunc(artsGpu->stream, artsGpuUnschedule, (void*)artsGpu));
    CHECKCORRECT(cudaLaunchHostFunc(artsGpu->stream, artsWrapUp, hostClosure));
#else
    // CHECKCORRECT(cudaStreamAddCallback(artsGpu->stream, artsGpuUnschedule, (void*)artsGpu, 0));
    CHECKCORRECT(cudaStreamAddCallback(artsGpu->stream, artsWrapUp, hostClosure, 0));
#endif
}

void artsScheduleToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr, artsGpu_t * artsGpu)
{
    artsGpuEdt_t * edt = (artsGpuEdt_t *)edtPtr;
    artsScheduleToGpuInternal(fnPtr, paramc, paramv, depc, depv, edt->grid, edt->block, edtPtr, artsGpu);
}

void artsGpuSynchronize(artsGpu_t * artsGpu)
{
    CHECKCORRECT(cudaStreamSynchronize(artsGpu->stream));
}

void artsGpuStreamBusy(artsGpu_t* artsGpu)
{
    CHECKCORRECT(cudaStreamQuery(artsGpu->stream));
}

//TODO: Make this more intelligent with memory usage (memUtil)
int artsGpuLookUp(unsigned id)
{
    // Loop over all devices to find available GPUs and return free artsGpu_t
    int start = (int) id % artsNumGpus;
    for (int i=start, j=0; j<artsNumGpus; ++j, i=(i+1)%artsNumGpus)
        if(artsAtomicFetchAdd(&artsGpus[i].scheduled, 1U))
            return i;
    return -1;
}

artsGpu_t * artsFindGpu(void * edtPacket, unsigned seed)
{
    artsGpu_t * ret = NULL;

    artsGpuEdt_t * edt = (artsGpuEdt_t *) edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    uint64_t maskOr=0, maskAnd=0, mask;
    for (unsigned int i=0; i<depc; ++i)
    {
        mask = artsGpuLookupDb(depv[i].guid);
        maskAnd &= mask;
        maskOr |= mask;
    }
    DPRINTF("MaskAnd: %p\n", maskAnd);
    DPRINTF("MaskOr: %p\n", maskOr);

    int gpu = -1;
    if (maskAnd) // All DBs in GPU
        gpu = __builtin_ctz(maskAnd);
    else if (maskOr) // At least one DB in GPU
        gpu = __builtin_ctz(maskOr);
    else
        gpu = artsGpuLookUp(seed);

    DPRINTF("Choosing gpu: %d\n", gpu);
    if(gpu > -1 && gpu < artsNumGpus)
        ret = &artsGpus[gpu];
    return ret;
}

void freeGpuItem(artsRouteItem_t * item)
{
    artsType_t type = artsGuidGetType(item->key);
    artsItemWrapper_t * wrapper = (artsItemWrapper_t*) item->data;
    if(type == ARTS_GPU_EDT)
    {
        artsGpuCleanUp_t * hostGCPtr = (artsGpuCleanUp_t *) wrapper->realData;
        DPRINTF("FREEING DEV PTR: %p\n", hostGCPtr->devClosure);
        CHECKCORRECT(cudaFree(hostGCPtr->devClosure));
        DPRINTF("FREEING HOST PTR: %p\n", hostGCPtr);
        CHECKCORRECT(cudaFreeHost(hostGCPtr));
    }
    else if(type > ARTS_BUFFER && type < ARTS_LAST_TYPE)  //DBs
        CHECKCORRECT(cudaFree(wrapper->realData));

    wrapper->realData = NULL;
    item->key = 0;
    item->lock = 0;
}