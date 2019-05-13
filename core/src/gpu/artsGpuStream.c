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

__thread int artsNumDevices = 0;
__thread artsGpuStream_t *artsStream;

// __thread volatile unsigned int deleteLock = 0;   //Per Device
// __thread artsArrayList * deleteQueue = NULL;     //Per Device
// __thread artsArrayList * deleteHostQueue = NULL; //Per Device

__thread artsGpuResCleanUp_t *myGpuResClean = NULL; // thread specific GPU closure

__thread volatile unsigned int newEdtLock = 0; //Can be shared across streams and devices
__thread artsArrayList * newEdts = NULL; //Can be shared across streams and devices

void artsInitGpuStream()
{
    CHECKCORRECT(cudaGetDeviceCount(&artsNumDevices));
    DPRINTF("NUM DEV: %d\n", artsNumDevices);
    artsStream = (artsGpuStream_t *)malloc(sizeof(artsGpuStream_t) * artsNumDevices);
    myGpuResClean = (artsGpuResCleanUp_t *)malloc(sizeof(artsGpuResCleanUp_t) * artsNumDevices);
    int i, j;
    for ( i = 0; i< artsNumDevices; i++) {
        CHECKCORRECT(cudaSetDevice(i));
        for (j = 0; j < MAX_STREAMS; j++)
            CHECKCORRECT(cudaStreamCreate(&(artsStream[i].stream[j])));
        artsLock(&(myGpuResClean[i].deleteLock));
        myGpuResClean[i].deleteQueue     = artsNewArrayList(sizeof(void*), 32);
        myGpuResClean[i].deleteHostQueue = artsNewArrayList(sizeof(void*), 32);
        artsUnlock(&(myGpuResClean[i].deleteLock));
    }
    // artsLock(&deleteLock);
    // deleteQueue     = artsNewArrayList(sizeof(void*), 32);
    // deleteHostQueue = artsNewArrayList(sizeof(void*), 32);
    // artsUnlock(&deleteLock);

    // thread level?
    artsLock(&newEdtLock);
    newEdts         = artsNewArrayList(sizeof(void*), 32); // should this be more?
    artsUnlock(&newEdtLock);
}

__thread volatile unsigned int * tlNewEdtLock;
__thread artsArrayList * tlNewEdts;
// Store each edts in a thread level edt queue
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
    int i, j;
    for(i = 0; i < artsNumDevices; i++) {
        CHECKCORRECT(cudaSetDevice(i));
        for(j=0; j < MAX_STREAMS; j++) {
            CHECKCORRECT(cudaStreamSynchronize(artsStream[i].stream[j]));
            CHECKCORRECT(cudaStreamDestroy(artsStream[i].stream[j]));
        }
    }
}

void CUDART_CB artsWrapUp(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpuCleanUp_t * gc = (artsGpuCleanUp_t*) data;
    //Shouldn't have to touch newly ready edts regardless of streams and devices
    struct artsGpuEdt * edt = (struct artsGpuEdt *) gc->edt;
    if(gc->edt)
    {
        tlNewEdtLock = gc->newEdtLock;
        tlNewEdts    = gc->newEdts;
        artsGpuHostWrapUp(gc->edt, edt->endGuid, edt->slot, edt->dataGuid);
    }

    //This should change for multi devices
    artsLock(gc->deleteLock);
    artsPushToArrayList(gc->deleteQueue,     &gc->devDB);
    artsPushToArrayList(gc->deleteQueue,     &gc->devClosure);
    artsPushToArrayList(gc->deleteHostQueue, &gc);
    artsUnlock(gc->deleteLock);
    DPRINTF("FINISHED GPU CALLS %s\n", cudaGetErrorString(status));
}

void artsScheduleToStreamInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr, int *strmIds, int strmCnt, int devId)
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
#ifdef DEBUG
    assert(strmCnt == depc);
    assert(strmCnt <= MAX_STREAMSi);
#endif

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

    CHECKCORRECT(cudaSetDevice(devId));
    //Fill host closure using last stream from available ones : default
    // this needs to change
    hostGCPtr->scheduleCounter = &artsStream[devId].scheduled[0];
    // hostGCPtr->deleteLock = &deleteLock;
    hostGCPtr->deleteLock = &myGpuResClean[devId].deleteLock;
    hostGCPtr->newEdtLock = &newEdtLock;
    // hostGCPtr->deleteQueue = deleteQueue;
    hostGCPtr->deleteQueue = myGpuResClean[devId].deleteQueue;
    hostGCPtr->deleteHostQueue = myGpuResClean[devId].deleteHostQueue;
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
    for(unsigned int i=0, s = 0; i<depc, s<strmCnt; i++, s++)
    {
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            //hostDepv now holds devDb + offset
            // CHECKCORRECT(cudaMemcpyAsync(hostDepv[i].ptr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsStream.stream));
            CHECKCORRECT(cudaMemcpyAsync(hostDepv[i].ptr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsStream[devId].stream[strmIds[s]]));
        }
    }

    DPRINTF("Filled GPU DBs\n");

    //Fill GPU closure (we don't need the edt which is why we start at hostParmv
    // CHECKCORRECT(cudaMemcpyAsync(devClosure, (void*)hostParamv, devClosureSize, cudaMemcpyHostToDevice, artsStream.stream));
    CHECKCORRECT(cudaMemcpyAsync(devClosure, (void*)hostParamv, devClosureSize, cudaMemcpyHostToDevice, artsStream[devId].stream[strmIds[strmCnt - 1]]));

    DPRINTF("Filled GPU Closure\n");

    //Launch kernel
    // fnPtr<<<grid, block, 0, artsStream.stream>>>(paramc, devParamv, depc, devDepv);
    fnPtr<<<grid, block, 0, artsStream[devId].stream[strmIds[strmCnt - 1]]>>>(paramc, devParamv, depc, devDepv);

    //Move data back
    unsigned int i,s;
    for(i=0, s=0; i<depc, s<strmCnt; i++, s++)
    {
        if(depv[i].ptr)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            // CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, artsStream.stream));
            CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, artsStream[devId].stream[s]));
        }
    }

    // CHECKCORRECT(cudaStreamAddCallback(artsStream.stream, artsWrapUp, hostClosure, 0));
    CHECKCORRECT(cudaStreamAddCallback(artsStream[devId].stream[strmIds[strmCnt - 1]], artsWrapUp, hostClosure, 0));
}

void artsScheduleToStream(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr)
{
    // all streams of device 0
    // This needs to be fixed..
    int * strmIds = (int*) malloc(sizeof(int) * MAX_STREAMS);
    for (unsigned int i = 0; i < MAX_STREAMS; i++)
        strmIds[i] = i;
    struct artsGpuEdt * edt = (struct artsGpuEdt *)edtPtr;
    artsScheduleToStreamInternal(fnPtr, paramc, paramv, depc, depv, edt->grid, edt->block, edtPtr, strmIds, MAX_STREAMS, 0);
}

// TODO:
// Default synchronization on strm 0 for which device?
void artsWaitForStream()
{
    CHECKCORRECT(cudaStreamSynchronize(artsStream[0].stream[0]));
}

// Synchronize subset of strms
void artsWaitForStream(int *strmIds, int strmCnt, int deviceId)
{
    CHECKCORRECT(cudaSetDevice(deviceId));
    for (unsigned int i = 0; i < strmCnt; i++) {
        CHECKCORRECT(cudaStreamSynchronize(artsStream[deviceId].stream[i]));
    }
}

// Default?
void artsStreamBusy()
{
    CHECKCORRECT(cudaStreamQuery(artsStream[0].stream[0]));
}

void artsStreamBusy(cudaStream_t strm)
{
    CHECKCORRECT(cudaStreamQuery(strm));
}

// unsigned int artsStreamScheduled()
// {
//     return artsStream.scheduled;
// }

unsigned int artsStreamScheduled(int strmId, int devId)
{
    return artsStream[devId].scheduled[strmId];
}

__thread uint64_t  devSize = 0; //Per device
__thread uint64_t hostSize = 0; //Per device

/* This is some really crappy problem
 * CudaFree can't run at the same time as the host callback, artsWrapUp.
 * Host callback can lock arrayList if the free already has lock.
 * Thus deadlock!  Instead we do this fancy split.
 */
void artsFreeGpuMemory()
{
    uint64_t  oldDevSize = devSize;
    uint64_t oldHostSize = hostSize;

    for(unsigned int d = 0; d<artsNumDevices; d++) {

      // if(artsTryLock(&deleteLock))
      if(artsTryLock(&myGpuResClean[d].deleteLock))
      {
          devSize   = artsLengthArrayList(myGpuResClean[d].deleteQueue);
          hostSize  = artsLengthArrayList(myGpuResClean[d].deleteHostQueue);
          artsUnlock(&myGpuResClean[d].deleteLock);
      }

      for(uint64_t i=oldDevSize; i<devSize; i++)
      {
          void ** ptr = (void**)artsGetFromArrayList(myGpuResClean[d].deleteQueue, i);
          CHECKCORRECT(cudaFree(*ptr));
      }

      for(uint64_t i=oldHostSize; i<hostSize; i++)
      {
          void ** ptr = (void**)artsGetFromArrayList(myGpuResClean[d].deleteHostQueue, i);
          CHECKCORRECT(cudaFreeHost(*ptr));
      }

      if(artsTryLock(&myGpuResClean[d].deleteLock))
      {
          if(devSize && artsLengthArrayList(myGpuResClean[d].deleteQueue) == devSize)
          {
              artsResetArrayList(myGpuResClean[d].deleteQueue);
              devSize = 0;

          }
          if(hostSize && artsLengthArrayList(myGpuResClean[d].deleteHostQueue) == hostSize)
          {
              artsResetArrayList(myGpuResClean[d].deleteHostQueue);
              hostSize = 0;
          }
          artsUnlock(&myGpuResClean[d].deleteLock);
      }
    }
}
