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

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

__thread artsGpuStream_t artsStream;
__thread volatile unsigned int deleteLock = 0;
__thread artsArrayList * deleteQueue = NULL;
__thread artsArrayList * deleteHostQueue = NULL;

void artsInitGpuStream(artsGpuStream_t * aStream)
{
    CHECKCORRECT(cudaStreamCreate(&aStream->stream));
    artsLock(&deleteLock);
    if(!deleteQueue)
        deleteQueue = artsNewArrayList(sizeof(void*), 32);
    if(!deleteHostQueue)
        deleteHostQueue = artsNewArrayList(sizeof(void*), 32);
    artsUnlock(&deleteLock);
}

void artsDestroyGpuStream(artsGpuStream_t * aStream)
{
    CHECKCORRECT(cudaStreamSynchronize(aStream->stream));
    CHECKCORRECT(cudaStreamDestroy(aStream->stream));
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpuCleanUp_t * gc = (artsGpuCleanUp_t*) data;    
//    artsGpuHostWrapUp(gc->edt);
    
    artsLock(gc->deleteLock);
    artsPushToArrayList(gc->deleteQueue,     &gc->devDB);
    artsPushToArrayList(gc->deleteQueue,     &gc->devClosure);
    artsPushToArrayList(gc->deleteHostQueue, &gc);
    artsUnlock(gc->deleteLock);
    
    PRINTF("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
}

void artsScheduleToStream(artsGpuStream_t * aStream, artsGpu_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, artsEdt * edtPtr)
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
        }
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
        if(paramc)
            hostParamv = (uint64_t*)(hostGCPtr + 1);
        if(depc)
            hostDepv = (artsEdtDep_t *)(hostParamv + paramc);
    }
    
    DPRINTF("devDB: %p devClosure: %p hostClosure: %p\n", devDB, devClosure, hostClosure);
    
    //Fill host closure
    hostGCPtr->deleteLock = &deleteLock;
    hostGCPtr->deleteQueue = deleteQueue;
    hostGCPtr->deleteHostQueue = deleteHostQueue;
    hostGCPtr->devDB = devDB;
    hostGCPtr->devClosure = devClosure;
    hostGCPtr->edt = edtPtr;
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
    
    //Fill GPU DBs
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
        struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
        size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
        //hostDepv now holds devDb + offset
        CHECKCORRECT(cudaMemcpyAsync(hostDepv[i].ptr, depv[i].ptr, size, cudaMemcpyHostToDevice, aStream->stream));
        }
    }
    
    //Fill GPU closure (we don't need the edt which is why we start at hostParmv
    CHECKCORRECT(cudaMemcpyAsync(devClosure, (void*)hostParamv, devClosureSize, cudaMemcpyHostToDevice, aStream->stream));
    
    //Launch kernel
    fnPtr<<<grid, block, 0, aStream->stream>>>(paramc, devParamv, depc, devDepv);
    
    //Move data back
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr)
        {
        struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
        size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
        CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, aStream->stream));
        }
    }
    
    CHECKCORRECT(cudaStreamAddCallback(aStream->stream, artsWrapUp, hostClosure, 0));
}

void artsWaitForStream(artsGpuStream_t * aStream)
{
    CHECKCORRECT(cudaStreamSynchronize(aStream->stream));
}

void artsStreamBusy(artsGpuStream_t * aStream)
{
    CHECKCORRECT(cudaStreamQuery(aStream->stream));
}

void artsFreeGpuMemory()
{
    artsLock(&deleteLock);
    
    uint64_t size = artsLengthArrayList(deleteQueue);
    DPRINTF("deleteQueue Size: %u\n", size);
    if(size)
    {
        for(uint64_t i=0; i<size; i++)
        {
            void ** ptr = (void**)artsGetFromArrayList(deleteQueue, i);
            DPRINTF("i: %u %p\n", i, *ptr);
            CHECKCORRECT(cudaFree(*ptr));
        }
        artsResetArrayList(deleteQueue);
    }
    
    size = artsLengthArrayList(deleteHostQueue);
    DPRINTF("deleteHostQueue Size: %u\n", size);
    if(size)
    {
        for(uint64_t i=0; i<size; i++)
        {
            void ** ptr = (void**)artsGetFromArrayList(deleteHostQueue, i);
            DPRINTF("Host i: %u %p\n", i, *ptr);
            CHECKCORRECT(cudaFreeHost(*ptr));
        }
        artsResetArrayList(deleteHostQueue);
    }
    
    artsUnlock(&deleteLock);
}
