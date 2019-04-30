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
#include "artsGpuStream.h"
#include "artsGpuRuntime.h"
#include "artsGlobals.h"
#include "artsDeque.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

__thread artsGpuStream_t artsStream;
__thread volatile unsigned int deleteLock = 0;
__thread volatile unsigned int newEdtLock = 0;
__thread artsArrayList * deleteQueue = NULL;
__thread artsArrayList * deleteHostQueue = NULL;
__thread artsArrayList * newEdts = NULL;

void artsInitGpuStream()
{
    CHECKCORRECT(cudaStreamCreate(&artsStream.stream));
    artsLock(&deleteLock);
    deleteQueue     = artsNewArrayList(sizeof(void*), 32);
    deleteHostQueue = artsNewArrayList(sizeof(void*), 32);
    artsUnlock(&deleteLock);
    
    artsLock(&newEdtLock);
    newEdts         = artsNewArrayList(sizeof(void*), 32);
    artsUnlock(&newEdtLock);
}

__thread volatile unsigned int * tlNewEdtLock;
__thread artsArrayList * tlNewEdts;

void artsStoreNewEdts(void * edt)
{
    artsLock(tlNewEdtLock);
    artsPushToArrayList(tlNewEdts, &edt);
    artsUnlock(tlNewEdtLock);
}

void artsHandleNewEdts()
{
    artsLock(&newEdtLock);
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
    artsUnlock(&newEdtLock);
}

void artsDestroyGpuStream()
{
    CHECKCORRECT(cudaStreamSynchronize(artsStream.stream));
    CHECKCORRECT(cudaStreamDestroy(artsStream.stream));
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpuCleanUp_t * gc = (artsGpuCleanUp_t*) data;
    struct artsGpuEdt * edt = (struct artsGpuEdt *) gc->edt;
    if(gc->edt)
    {
        tlNewEdtLock = gc->newEdtLock;
        tlNewEdts    = gc->newEdts;
        artsGpuHostWrapUp(gc->edt, edt->endGuid, edt->slot, edt->dataGuid);
    }
    
    artsLock(gc->deleteLock);
    artsPushToArrayList(gc->deleteQueue,     &gc->devDB);
    artsPushToArrayList(gc->deleteQueue,     &gc->devClosure);
    artsPushToArrayList(gc->deleteHostQueue, &gc);
    artsUnlock(gc->deleteLock);
    DPRINTF("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
}

void artsScheduleToStreamInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr)
{
//    For now this should push the following into the stream:
//    1. Copy data from host to device
//    2. Push kernel
//    3. Copy data from device to host
//    4. Call host callback function artsGpuHostWrapUp
    
    void * devDB       = NULL;
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
    
    //Get size of DBs
    uint64_t totalDBSize = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            totalDBSize += db->header.size - sizeof(struct artsDb);
            DPRINTF("SIZE[%u]: %u\n", i, db->header.size - sizeof(struct artsDb));
        }
        else
            DPRINTF("SIZE[%u]: %p\n", i, depv[i].ptr);
    }
    
    DPRINTF("totalDBSize: %u devClosureSize: %u hostClosureSize: %u\n", totalDBSize, devClosureSize, hostClosureSize);
    
    //Allocate space for DB on GPU
    if(totalDBSize)
        CHECKCORRECT(cudaMalloc(&devDB, totalDBSize));
    
    //Allocate Closure for GPU
    if(devClosureSize)
    {
        CHECKCORRECT(cudaMalloc(&devClosure, devClosureSize));
        devParamv = (uint64_t*) devClosure;
        devDepv = (artsEdtDep_t *)(devParamv + paramc);
    }
    
    //Allocate closure for host
    if(hostClosureSize)
    {
        CHECKCORRECT(cudaMallocHost(&hostClosure, hostClosureSize));
        hostGCPtr = (artsGpuCleanUp_t *) hostClosure;
        hostParamv = (uint64_t*)(hostGCPtr + 1);
        hostDepv = (artsEdtDep_t *)(hostParamv + paramc);
    }
    
    DPRINTF("devDB: %p devClosure: %p hostClosure: %p\n", devDB, devClosure, hostClosure);
    
    //Fill host closure
    hostGCPtr->deleteLock = &deleteLock;
    hostGCPtr->newEdtLock = &newEdtLock;
    hostGCPtr->deleteQueue = deleteQueue;
    hostGCPtr->deleteHostQueue = deleteHostQueue;
    hostGCPtr->newEdts = newEdts;
    hostGCPtr->devDB = devDB;
    hostGCPtr->devClosure = devClosure;
    hostGCPtr->edt = (struct artsEdt*)edtPtr;
    for(unsigned int i=0; i<paramc; i++)
        hostParamv[i] = paramv[i];
    uint64_t tempSize = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        uint64_t size = 0;
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size = db->header.size - sizeof(struct artsDb);
            hostDepv[i].ptr  = (char*)devDB + tempSize;
        }
        else
            hostDepv[i].ptr  = NULL;
        hostDepv[i].guid = depv[i].guid;
        hostDepv[i].mode = depv[i].mode;
        tempSize+=size;
    }
    
    DPRINTF("Filled host closure\n");
    
    //Fill GPU DBs
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            //hostDepv now holds devDb + offset
            CHECKCORRECT(cudaMemcpyAsync(hostDepv[i].ptr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsStream.stream));
        }
    }

    DPRINTF("Filled GPU DBs\n");
    
    //Fill GPU closure (we don't need the edt which is why we start at hostParmv
    CHECKCORRECT(cudaMemcpyAsync(devClosure, (void*)hostParamv, devClosureSize, cudaMemcpyHostToDevice, artsStream.stream));
    
    
    DPRINTF("Filled GPU Closure\n");
    
    //Launch kernel
    fnPtr<<<grid, block, 0, artsStream.stream>>>(paramc, devParamv, depc, devDepv);
    
    //Move data back
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, artsStream.stream));
        }
    }
    
    CHECKCORRECT(cudaStreamAddCallback(artsStream.stream, artsWrapUp, hostClosure, 0));
}

void artsScheduleToStream(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr)
{
    struct artsGpuEdt * edt = (struct artsGpuEdt *)edtPtr;
    artsScheduleToStreamInternal(fnPtr, paramc, paramv, depc, depv, edt->grid, edt->block, edtPtr);
}

void artsWaitForStream()
{
    CHECKCORRECT(cudaStreamSynchronize(artsStream.stream));
}

void artsStreamBusy()
{
    CHECKCORRECT(cudaStreamQuery(artsStream.stream));
}

__thread uint64_t  devSize = 0;
__thread uint64_t hostSize = 0;

/* This is some really crappy problem
 * CudaFree can't run at the same time as the host callback, artsWrapUp.
 * Host callback can lock arrayList if the free already has lock.
 * Thus deadlock!  Instead we do this fancy split.  
 */
void artsFreeGpuMemory()
{    
    uint64_t  oldDevSize = devSize;
    uint64_t oldHostSize = hostSize;
    
    if(artsTryLock(&deleteLock))
    {
        devSize   = artsLengthArrayList(deleteQueue);
        hostSize  = artsLengthArrayList(deleteHostQueue);
        artsUnlock(&deleteLock);
    }

    for(uint64_t i=oldDevSize; i<devSize; i++)
    {
        void ** ptr = (void**)artsGetFromArrayList(deleteQueue, i);
        CHECKCORRECT(cudaFree(*ptr));
    }

    for(uint64_t i=oldHostSize; i<hostSize; i++)
    {
        void ** ptr = (void**)artsGetFromArrayList(deleteHostQueue, i);
        CHECKCORRECT(cudaFreeHost(*ptr));
    }

    if(artsTryLock(&deleteLock))
    {
        if(devSize && artsLengthArrayList(deleteQueue) == devSize)
        {
            artsResetArrayList(deleteQueue);
            devSize = 0;
            
        }
        if(hostSize && artsLengthArrayList(deleteHostQueue) == hostSize)
        {
            artsResetArrayList(deleteHostQueue);
            hostSize = 0;
        }
        artsUnlock(&deleteLock);
    }
}
