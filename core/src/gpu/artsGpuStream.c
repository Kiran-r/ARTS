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
#include "artsRouteTable.h"
#include "artsDebug.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

int artsNumGpus = 0;
artsGpu_t * artsGpus;

volatile unsigned int newEdtLock = 0; //Can be shared across streams and devices
artsArrayList * newEdts = NULL; //Can be shared across streams and devices

volatile unsigned int * tlNewEdtLock;
artsArrayList * tlNewEdts;
//__thread artsGpu_t artsGpu;


static cudaError_t artsCreateStreams (artsGpu_t * artsGpu) {
  DPRINTF("Creating Stream on %d\n", artsGpu->device);
  CHECKCORRECT(cudaStreamCreate(&artsGpu->stream)); // Make it scalable
  DPRINTF("Created Stream on %d\n", artsGpu->device);
  artsGpu->deleteQueue     = artsNewArrayList(sizeof(void*), 32);
  artsGpu->deleteHostQueue = artsNewArrayList(sizeof(void*), 32);
  artsGpu->scheduled = 0U;
  return cudaSuccess;
}

void artsInitGpus(unsigned int entries, unsigned int tableSize, int numGpus)
{
    int savedDevice;
    artsGpu_t * artsGpu;
    //CHECKCORRECT(cudaGetDeviceCount(&artsNumGpus));
    artsNumGpus = numGpus;
    DPRINTF("NUM DEV: %d\n", artsNumGpus);
    artsGpus = (artsGpu_t*)malloc(sizeof(artsGpu_t)*artsNumGpus);
    cudaGetDevice(&savedDevice);

    // Initialize artsGpu with 1 stream/GPU
    for (int i=0; i<artsNumGpus; ++i)
    {
        artsGpu = artsGpus + i;
        artsGpu->device = i;
        DPRINTF("Setting %d\n", i);
        CHECKCORRECT(cudaSetDevice(i));
        CHECKCORRECT(artsCreateStreams(artsGpu));
        artsNodeInfo.gpuRouteTable[i] = artsRouteTableListNew(1, entries, tableSize);
    }

    artsLock(&newEdtLock);
    newEdts         = artsNewArrayList(sizeof(void*), 32);
    artsUnlock(&newEdtLock);
    cudaSetDevice(savedDevice);
}

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

void artsCleanupGpus()
{
    int savedDevice;
    cudaGetDevice(&savedDevice);
    for (int i=0; i<artsNumGpus; i++)
    {
        CHECKCORRECT(cudaSetDevice((artsGpus+i)->device));
        CHECKCORRECT(cudaStreamSynchronize((artsGpus+i)->stream));
        CHECKCORRECT(cudaStreamDestroy((artsGpus+i)->stream));
    }
    cudaSetDevice(savedDevice);
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

void artsScheduleToGpuInternal(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, dim3 grid, dim3 block, void * edtPtr, artsGpu_t * artsGpu)
{
//    For now this should push the following into the stream:
//    1. Copy data from host to device
//    2. Push kernel
//    3. Copy data from device to host
//    4. Call host callback function artsGpuHostWrapUp

    static volatile unsigned int Gpulock;

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

    artsLock(&Gpulock);
    cudaSetDevice(artsGpu->device);
    artsUnlock(&Gpulock);

    if(hostClosureSize)
    {
        //Allocate closure for host
        CHECKCORRECT(cudaMallocHost(&hostClosure, hostClosureSize));
        hostGCPtr = (artsGpuCleanUp_t *) hostClosure;
        hostParamv = (uint64_t*)(hostGCPtr + 1);
        hostDepv = (artsEdtDep_t *)(hostParamv + paramc);

        //Fill Host closure
        hostGCPtr->scheduleCounter = &artsGpu->scheduled;
        hostGCPtr->deleteLock = &artsGpu->deleteLock;
        hostGCPtr->newEdtLock = &newEdtLock;
        hostGCPtr->deleteQueue = artsGpu->deleteQueue;
        hostGCPtr->deleteHostQueue = artsGpu->deleteHostQueue;
        hostGCPtr->newEdts = newEdts;
        hostGCPtr->devDB = devDB;
        hostGCPtr->devClosure = devClosure;
        hostGCPtr->edt = (struct artsEdt*)edtPtr;
        for(unsigned int i=0; i<paramc; i++)
            hostParamv[i] = paramv[i];
    
        DPRINTF("Filled host closure\n");
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

                bool ret = artsGpuRouteTableAddItemRace(dataPtr, depv[i].guid, artsGlobalRankId, artsGpu->device);
                if (!ret) //Someone beat us to creating the data... So we must free
                    cudaFree(dataPtr);
                else //We won, so move data
                {
                    //hostDepv now holds devDb + offset
                    CHECKCORRECT(cudaMemcpyAsync(dataPtr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsGpu->stream));

                }
            }
            hostDepv[i].ptr = dataPtr;
        }
        else
            hostDepv[i].ptr = NULL;

        hostDepv[i].guid = depv[i].guid;
        hostDepv[i].mode = depv[i].mode;    
    }

    //Allocate Closure for GPU
    if(devClosureSize)
    {
        CHECKCORRECT(cudaMalloc(&devClosure, devClosureSize));
        devParamv = (uint64_t*) devClosure;
        devDepv = (artsEdtDep_t *)(devParamv + paramc);
    }
    
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
        if(artsAtomicSchedule(&(artsGpus+i)->scheduled))
            return i;
    return -1;
}

artsGpu_t * artsFindGpu(void * edtPacket, unsigned seed)
{
    artsGpu_t * ret = NULL;

    struct artsGpuEdt * edt = (struct artsGpuEdt *) edtPacket;
    uint32_t       paramc = edt->wrapperEdt.paramc;
    uint32_t       depc   = edt->wrapperEdt.depc;
    uint64_t     * paramv = (uint64_t *)(edt + 1);
    artsEdtDep_t * depv   = (artsEdtDep_t *)(paramv + paramc);

    uint64_t maskOr=0, maskAnd=0, mask;
    for (unsigned int i=0; i<depc; ++i)
    {
        mask = artsLookupGpuDb(depv[i].guid);
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
    if(gpu >= artsNumGpus)
        artsDebugGenerateSegFault();

    if(gpu > -1)
        ret = artsGpus + gpu;
    return ret;
}

/* This is some really crappy problem
 * CudaFree can't run at the same time as the host callback, artsWrapUp.
 * Host callback can lock arrayList if the free already has lock.
 * Thus deadlock!  Instead we do this fancy split.  
 */
void artsFreeGpuMemory(artsGpu_t * artsGpu)
{
    uint64_t*  devSize = &artsGpu->devSize;
    uint64_t* hostSize = &artsGpu->hostSize;

    uint64_t  oldDevSize = *devSize;
    uint64_t oldHostSize = *hostSize;

    cudaSetDevice(artsGpu->device);
    *devSize   = artsLengthArrayList(artsGpu->deleteQueue);
    *hostSize  = artsLengthArrayList(artsGpu->deleteHostQueue);

    for(uint64_t i=oldDevSize; i<*devSize; i++)
    {
        void ** ptr = (void**)artsGetFromArrayList(artsGpu->deleteQueue, i);
        if (ptr)
            CHECKCORRECT(cudaFree(*ptr));
    }

    for(uint64_t i=oldHostSize; i<*hostSize; i++)
    {
        void ** ptr = (void**)artsGetFromArrayList(artsGpu->deleteHostQueue, i);
        if (ptr)
            CHECKCORRECT(cudaFreeHost(*ptr));
    }

    if(*devSize && artsLengthArrayList(artsGpu->deleteQueue) == *devSize)
    {
        artsResetArrayList(artsGpu->deleteQueue);
        *devSize = 0;
    }
    if(*hostSize && artsLengthArrayList(artsGpu->deleteHostQueue) == *hostSize)
    {
        artsResetArrayList(artsGpu->deleteHostQueue);
        *hostSize = 0;
    }
}

void artsGpuFree(void * data, unsigned int gpu)
{
    int savedDevice;
    cudaGetDevice(&savedDevice);
    cudaSetDevice((int)gpu);
    cudaFree(data);
    cudaSetDevice(savedDevice);
}

//------------------------------------------------Kiran-------------------------------------------------------------//
void artsScheduleToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, void * edtPtr, artsGpu_t * artsGpu)
{
    struct artsGpuEdt * edt = (struct artsGpuEdt *)edtPtr;
    artsScheduleToGpuInternal(fnPtr, paramc, paramv, depc, depv, edt->grid, edt->block, edtPtr, artsGpu);
}

void CUDART_CB artsGpuUnschedule(cudaStream_t stream, cudaError_t status, void * data)
{
    artsGpu_t * artsGpu = (artsGpu_t *) data;
    artsAtomicUnschedule(&artsGpu->scheduled);


}

artsGpu_t * artsGpuScheduled(unsigned int seed)
{
    int ret = artsGpuLookUp(seed);
    if (ret != -1)
        return artsGpus+ret;
    return NULL;
}

void ** artsGpuHostToDeviceDbs (uint32_t depc, uint32_t paramc, uint64_t * paramv, artsEdtDep_t * depv, int gpuId, artsGuid_t edtGuid, artsGpu_t * artsGpu, void ** devParamv)
{
    //Gets the dbs that need to written
    void ** writeDbs = NULL;
    uint64_t numWriteDbs = 0;
    for (unsigned int i=0; i<depc; ++i)
        if(depv[i].ptr && depv[i].mode == ARTS_DB_GPU_WRITE)
            numWriteDbs++;

    writeDbs = (void**) artsCalloc(sizeof(void*) * numWriteDbs);
    int index = 0;
    for (unsigned int i=0; i<depc; ++i)
        if(depv[i].ptr)
        {
            void * dataPtr = artsGpuRouteTableLookupDb(depv[i].guid, artsGpu->device);
            if (!dataPtr)
            {
                struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
                size_t size = db->header.size - sizeof(struct artsDb);
                CHECKCORRECT(cudaMalloc(&dataPtr, size));
                bool ret = artsGpuRouteTableAddItemRace(dataPtr, depv[i].guid, artsGlobalRankId, artsGpu->device);
                if (!ret)
                    cudaFree(dataPtr);
                else
                    CHECKCORRECT(cudaMemcpyAsync(dataPtr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsGpu->stream));
            }

            if (depv[i].mode == ARTS_DB_GPU_WRITE)
            {
                writeDbs[index] = depv[i].ptr;
                index++;
            }

            depv[i].ptr = dataPtr;
        }

    size_t  paramSize = sizeof(uint64_t) * paramc + sizeof(artsEdtDep_t) * depc;
    CHECKCORRECT(cudaMalloc(&devParamv, paramSize));
    CHECKCORRECT(cudaMemcpyAsync(devParamv, paramv, paramSize, cudaMemcpyHostToDevice, artsGpu->stream));
    artsGpuRouteTableAddItemRace(devParamv, edtGuid, artsGlobalRankId, gpuId);

    return writeDbs;
}


void artsScheduleKernelToGpu(artsEdt_t fnPtr, uint32_t paramc, uint64_t * gpuParamv, uint32_t depc, artsEdtDep_t * gpuDepv, dim3 grid, dim3 block, artsGpu_t * artsGpu)
{
    fnPtr<<<grid, block, 0, artsGpu->stream>>>(paramc, gpuParamv, depc, gpuDepv);
}

void artsGpuDeviceToHostDbs (uint32_t paramc,  uint32_t depc, artsEdtDep_t * depv, artsEdtDep_t * devDepv, artsGpu_t * artsGpu, void ** writeDbs)
{
    unsigned int index = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr && depv[i].mode == ARTS_DB_GPU_WRITE)
        {
            struct artsDb * db = (struct artsDb *) writeDbs[index] - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            CHECKCORRECT(cudaMemcpyAsync(writeDbs[index], depv[i].ptr, size, cudaMemcpyDeviceToHost, artsGpu->stream));
            index++;
        }
        artsGpuRouteTableReturnDb(depv[i].guid, true, artsGpu->device);
    }

#if CUDART_VERSION >= 10000
    CHECKCORRECT(cudaLaunchHostFunc(artsGpu->stream, artsGpuUnschedule, (void*)artsGpu));
#else
    CHECKCORRECT(cudaStreamAddCallback(artsGpu->stream, artsGpuUnschedule, (void*)artsGpu, 0));
#endif
}