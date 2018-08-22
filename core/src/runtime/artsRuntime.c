#include <unistd.h>
#include "arts.h"
#include "artsMalloc.h"
#include "artsDeque.h"
#include "artsAtomics.h"
#include "artsGuid.h"
#include "artsCounter.h"
#include "artsGlobals.h"
#include "artsRemote.h"
#include "artsOutOfOrder.h"
#include "artsRuntime.h"
#include <inttypes.h>
#include "artsCounter.h"
#include "artsDebug.h"
#include "artsRemoteFunctions.h"
#include "artsDbFunctions.h"
#include "artsAbstractMachineModel.h"
#include "artsRouteTable.h"
#include "artsEdtFunctions.h"
#include "artsEventFunctions.h"
#include "artsThreads.h"
#include "artsOutOfOrder.h"
#include "artsIntrospection.h"
#include "artsTerminationDetection.h"
#include <stdlib.h>
#include <stdio.h>
#include "artsTimer.h"
#include "artsArrayList.h"
#include "artsQueue.h"

//#define SIM_INTEL

#define DPRINTF( ... )
#define PACKET_SIZE 4096
#define NETWORK_BACKOFF_INCREMENT 0

extern unsigned int numNumaDomains;
extern int mainArgc;
extern char **mainArgv;
extern void initPerNode(unsigned int nodeId, int argc, char** argv) __attribute__((weak));
extern void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char**argv) __attribute__((weak));
extern void artsMain(int argc, char** argv) __attribute__((weak));

struct artsRuntimeShared artsNodeInfo;
__thread struct artsRuntimePrivate artsThreadInfo;

typedef bool (*scheduler_t)();
scheduler_t schedulerLoop[3] = {artsDefaultSchedulerLoop, artsNetworkBeforeStealSchedulerLoop, artsNetworkFirstSchedulerLoop};

artsGuid_t artsMainEdt(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    artsMain(mainArgc, mainArgv);
    return NULL_GUID;
}

void artsRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct artsConfig * config)
{
    artsThreadSetOsThreadCount(config->osThreadCount);
    artsNodeInfo.scheduler = schedulerLoop[config->scheduler];
    artsNodeInfo.deque = (struct artsDeque**) artsMalloc(sizeof(struct artsDeque*)*totalThreads);
    artsNodeInfo.receiverDeque = (struct artsDeque**) artsMalloc(sizeof(struct artsDeque*)*receiverThreads);
    artsNodeInfo.numaQueue = (struct artsQueue**) artsMalloc(sizeof(struct artsQueue*)*numNumaDomains);
    for(unsigned int i=0; i<numNumaDomains; i++)
        artsNodeInfo.numaQueue[i] = artsNewQueue();
    artsNodeInfo.routeTable = (struct artsRouteTable**) artsCalloc(sizeof(struct artsRouteTable*)*totalThreads);
    artsNodeInfo.remoteRouteTable = artsRouteTableListNew(1, config->routeTableEntries, config->routeTableSize);
    artsNodeInfo.localSpin = (volatile bool**) artsCalloc(sizeof(bool*)*totalThreads);
    artsNodeInfo.memoryMoves = (unsigned int**) artsCalloc(sizeof(unsigned int*)*totalThreads);
    artsNodeInfo.atomicWaits = (struct atomicCreateBarrierInfo **) artsCalloc(sizeof(struct atomicCreateBarrierInfo*)*totalThreads);
    artsNodeInfo.workerThreadCount = workerThreads;
    artsNodeInfo.senderThreadCount = senderThreads;
    artsNodeInfo.receiverThreadCount = receiverThreads;
    artsNodeInfo.totalThreadCount       = totalThreads;
    artsNodeInfo.readyToPush            = totalThreads;
    artsNodeInfo.readyToParallelStart   = totalThreads;
    artsNodeInfo.readyToInspect         = totalThreads;
    artsNodeInfo.readyToExecute         = totalThreads;
    artsNodeInfo.readyToClean           = totalThreads;
    artsNodeInfo.sendLock = 0U;
    artsNodeInfo.recvLock = 0U;
    artsNodeInfo.shutdownCount = artsGlobalRankCount-1;
    artsNodeInfo.shutdownStarted=0;
    artsNodeInfo.readyToShutdown = artsGlobalRankCount-1;
    artsNodeInfo.stealRequestLock = !remoteStealingOn;
    artsNodeInfo.buf = artsMalloc( PACKET_SIZE );
    artsNodeInfo.packetSize = PACKET_SIZE;
    artsNodeInfo.printNodeStats = config->printNodeStats;
    artsNodeInfo.shutdownEpoch = (config->shutdownEpoch) ? 1 : NULL_GUID ;
    artsNodeInfo.shadLoopStride = config->shadLoopStride;
    artsInitIntrospector(config);
}

void artsRuntimeGlobalCleanup()
{
    artsIntrospectivePrintTotals(artsGlobalRankId);
    artsFree(artsNodeInfo.deque);
    artsFree((void *)artsNodeInfo.localSpin);
    artsFree(artsNodeInfo.memoryMoves);
    artsFree(artsNodeInfo.atomicWaits);
}

void artsThreadZeroNodeStart()
{
    artsStartInspector(1);

    setGlobalGuidOn();
    createShutdownEpoch();
    
    if(initPerNode)
        initPerNode(artsGlobalRankId, mainArgc, mainArgv);
    setGuidGeneratorAfterParallelStart();

    artsStartInspector(2);
    ARTSSTARTCOUNTING(2);
    artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
    while(artsNodeInfo.readyToParallelStart){ }
    if(initPerWorker && artsThreadInfo.worker)
        initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc, mainArgv);
    
    if(artsMain && !artsGlobalRankId)
        artsEdtCreate(artsMainEdt, 0, 0, NULL, 0);
    
    artsIncrementFinishedEpochList();
    
    artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
    while(artsNodeInfo.readyToInspect){ }
    ARTSSTARTCOUNTING(3);
    artsStartInspector(3);
    artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
    while(artsNodeInfo.readyToExecute){ }
}

void artsRuntimePrivateInit(struct threadMask * unit, struct artsConfig  * config)
{
    artsNodeInfo.deque[unit->id] = artsThreadInfo.myDeque = artsDequeNew(config->dequeSize);
    if(unit->worker)
    {
        artsNodeInfo.routeTable[unit->id] =  artsRouteTableListNew(1, config->routeTableEntries, config->routeTableSize);
    }

    if(unit->networkSend || unit->networkReceive)
    {
        if(unit->networkSend)
        {
            unsigned int size = artsGlobalRankCount*config->ports / artsNodeInfo.senderThreadCount;
            unsigned int rem = artsGlobalRankCount*config->ports % artsNodeInfo.senderThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                artsRemotSetThreadOutboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                artsRemotSetThreadOutboundQueues(start, start+size);
            }
        }
        if(unit->networkReceive)
        {
            artsNodeInfo.receiverDeque[unit->groupPos] = artsNodeInfo.deque[unit->id];
            unsigned int size = (artsGlobalRankCount-1)*config->ports / artsNodeInfo.receiverThreadCount;
            unsigned int rem = (artsGlobalRankCount-1)*config->ports % artsNodeInfo.receiverThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                artsRemotSetThreadInboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                artsRemotSetThreadInboundQueues(start, start+size);
            }

        }
    }
    artsNodeInfo.localSpin[unit->id] = &artsThreadInfo.alive;
    artsThreadInfo.alive = true;
    artsNodeInfo.memoryMoves[unit->id] = (unsigned int *)&artsThreadInfo.oustandingMemoryMoves;
    artsNodeInfo.atomicWaits[unit->id] = &artsThreadInfo.atomicWait;
    artsThreadInfo.atomicWait.wait = true;
    artsThreadInfo.oustandingMemoryMoves = 0;
    artsThreadInfo.coreId = unit->unitId;
    artsThreadInfo.threadId = unit->id;
    artsThreadInfo.groupId = unit->groupPos;
    artsThreadInfo.clusterId = unit->clusterId;
    artsThreadInfo.worker = unit->worker;
    artsThreadInfo.networkSend = unit->networkSend;
    artsThreadInfo.networkReceive = unit->networkReceive;
    artsThreadInfo.backOff = 1;
    artsThreadInfo.currentEdtGuid = 0;
    artsThreadInfo.mallocType = artsDefaultMemorySize;
    artsThreadInfo.mallocTrace = 1;
    artsThreadInfo.localCounting = 1;
    artsThreadInfo.shadLock = 0;
    artsGuidKeyGeneratorInit();
    INITCOUNTERLIST(unit->id, artsGlobalRankId, config->counterFolder, config->counterStartPoint);
    artsAtomicSub(&artsNodeInfo.readyToPush, 1U);
    while(artsNodeInfo.readyToPush){  };
    if(unit->id)
    {
        artsAtomicSub(&artsNodeInfo.readyToParallelStart, 1U);
        while(artsNodeInfo.readyToParallelStart){ };

        if(artsThreadInfo.worker)
        {
            if(initPerWorker)
                initPerWorker(artsGlobalRankId, artsThreadInfo.groupId, mainArgc, mainArgv);
            artsIncrementFinishedEpochList();
        }
        
        artsAtomicSub(&artsNodeInfo.readyToInspect, 1U);
        while(artsNodeInfo.readyToInspect) { };
        artsAtomicSub(&artsNodeInfo.readyToExecute, 1U);
        while(artsNodeInfo.readyToExecute) { };
    }
    artsThreadInfo.drand_buf[0] = 1202107158 + unit->id * 1999;
    artsThreadInfo.drand_buf[1] = 0;
    artsThreadInfo.drand_buf[2] = 0;
}

void artsRuntimePrivateCleanup()
{
    artsAtomicSub(&artsNodeInfo.readyToClean, 1U);
    while(artsNodeInfo.readyToClean){ };
    if(artsThreadInfo.myDeque)
        artsDequeDelete(artsThreadInfo.myDeque);
    if(artsThreadInfo.myNodeDeque)
        artsDequeDelete(artsThreadInfo.myNodeDeque);
#if defined(COUNT) || defined(MODELCOUNT)
    artsWriteCountersToFile(artsThreadInfo.threadId, artsGlobalRankId);
#endif
    artsWriteMetricShotFile(artsThreadInfo.threadId, artsGlobalRankId);
}

void artsRuntimeStop()
{
    unsigned int i;
    for(i=0; i<artsNodeInfo.totalThreadCount; i++)
    {
        while(!artsNodeInfo.localSpin[i]);
        (*artsNodeInfo.localSpin[i]) = false;
    }
    artsStopInspector();
}

void artsHandleRemoteStolenEdt(struct artsEdt *edt)
{
    DPRINTF("push stolen %d\n",artsThreadInfo.coreId);
    incrementQueueEpoch(edt->epochGuid);
    globalShutdownGuidIncQueue();
    artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
}

void artsHandleReadyEdt(struct artsEdt * edt)
{
    acquireDbs(edt);
    if(artsAtomicSub(&edt->depcNeeded,1U) == 0)
    {
        incrementQueueEpoch(edt->epochGuid);
        globalShutdownGuidIncQueue();
//        if(edt->cluster == artsThreadInfo.clusterId)
            artsDequePushFront(artsThreadInfo.myDeque, edt, 0);
//        else
//            enqueue((uint64_t)edt, artsNodeInfo.numaQueue[artsThreadInfo.clusterId]);
//        enqueue((uint64_t)edt, artsNodeInfo.numaQueue[edt->cluster]);
        artsUpdatePerformanceMetric(artsEdtQueue, artsThread, 1, false);
    }
}

static inline void artsRunEdt(void *edtPacket)
{
    struct artsEdt *edt = edtPacket;
    u32 depc = edt->depc;
    artsEdtDep_t * depv = (artsEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);

    artsEdt_t func = edt->funcPtr;
    u32 paramc = edt->paramc;
    u64 *paramv = (u64 *)(edt + 1);

    prepDbs(depc, depv);

    artsSetThreadLocalEdtInfo(edt);
    ARTSCOUNTERTIMERSTART(edtCounter);

    artsGuid_t result = func(paramc, paramv, depc, depv);

    ARTSCOUNTERTIMERENDINCREMENT(edtCounter);
    artsUpdatePerformanceMetric(artsEdtThroughput, artsThread, 1, false);

    artsUnsetThreadLocalEdtInfo();

    if(edt->outputEvent != NULL_GUID)
    {
        if(artsGuidGetType(edt->outputEvent) == ARTS_EVENT)
            artsEventSatisfySlot(edt->outputEvent, result, ARTS_EVENT_LATCH_DECR_SLOT);
        else //This is for a synchronous path
        {
            artsSetBuffer(edt->outputEvent, artsCalloc(sizeof(unsigned int)), sizeof(unsigned int));
        }
    }
    
    releaseDbs(depc, depv);
    artsEdtDelete(edtPacket);
}

inline unsigned int artsRuntimeStealAnyMultipleEdt( unsigned int amount, void ** returnList )
{
    struct artsEdt *edt = NULL;
    unsigned int i;
    unsigned int count = 0;
    bool done = false;
    for (i=0; i<artsNodeInfo.workerThreadCount && !done; i++)
    {
        do
        {
//            edt = artsDequePopBack(artsNodeInfo.workerDeque[i]);
            if(edt != NULL)
            {
                returnList[ count ] = edt;
                count++;
                if(count == amount)
                    done = true;
            }
        }while(edt != NULL && !done);
    }
    return count;
}

inline struct artsEdt * artsRuntimeStealFromNetwork()
{
    struct artsEdt *edt = NULL;
    if(artsGlobalRankCount > 1)
    {
        unsigned int index = artsThreadInfo.threadId;
        for (unsigned int i=0; i<artsNodeInfo.receiverThreadCount; i++)
        {
            index = (index + 1) % artsNodeInfo.receiverThreadCount;
            if(edt = artsDequePopBack(artsNodeInfo.receiverDeque[index]))
                break;
        }
    }
    return edt;
}

__thread unsigned int lastHitThread = 0;

inline struct artsEdt * artsCheckLastWorker() {
    struct artsEdt * ret = artsDequePopBack(artsNodeInfo.deque[lastHitThread]);
    if(ret)
        artsUpdatePerformanceMetric(artsEdtLastLocalHit, artsThread, 1, false);
    return ret;
}

inline struct artsEdt * artsRuntimeStealFromWorker()
{
    struct artsEdt *edt = NULL;
    if(artsNodeInfo.totalThreadCount > 1)
    {
        
        long unsigned int stealLoc;
        do
        {
            stealLoc = jrand48(artsThreadInfo.drand_buf);
            stealLoc = stealLoc % artsNodeInfo.totalThreadCount;
        } while(stealLoc == artsThreadInfo.threadId);

        edt = artsDequePopBack(artsNodeInfo.deque[stealLoc]);
        
        if(edt)
        {
            lastHitThread = (unsigned int) stealLoc;
        }
    }
    return edt;
}

struct artsEdt * artsRuntimeStealFromWorkerMult()
{
    struct artsEdt * edt = NULL;
    if(artsNodeInfo.totalThreadCount > 1)
    {
        
        long unsigned int stealLoc;
        do
        {
            stealLoc = jrand48(artsThreadInfo.drand_buf);
            stealLoc = stealLoc % artsNodeInfo.totalThreadCount;
        } while(stealLoc == artsThreadInfo.threadId);

        unsigned int size = artsDequeSize(artsNodeInfo.deque[stealLoc]);
        unsigned int half = size/2;
        unsigned int stolen = 0;
        if(STEALSIZE < half)
        {
            for(unsigned int i=0; i<half; i+=STEALSIZE) 
            {
                void ** array = artsDequePopBackMult(artsNodeInfo.deque[stealLoc]);
                if(array)
                {
                    for(unsigned int j=0; j<STEALSIZE; j++)
                    {
                        if(edt)
                            artsDequePushFront(artsThreadInfo.myDeque, array[j], 0);
                        else
                            edt = array[j];
                    }
                    stolen+=STEALSIZE;
                }
                else
                    break;
            }

            if(edt)
            {
                artsUpdatePerformanceMetric(artsEdtSteal, artsThread, stolen, false);
                lastHitThread = (unsigned int) stealLoc;
            }
        }
    }
    return edt;
}

bool artsNetworkFirstSchedulerLoop()
{
//    struct artsEdt *edtFound;
//    if(!(edtFound = artsRuntimeStealFromNetwork()))
//    {
//        if(!(edtFound = artsDequePopFront(artsThreadInfo.myNodeDeque)))
//        {
//            if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
//                edtFound = artsRuntimeStealFromWorker();
//        }
//    }
//    if(edtFound)
//    {
//        artsRunEdt(edtFound);
//        return true;
//    }
    return false;
}

bool artsNetworkBeforeStealSchedulerLoop()
{
//    struct artsEdt *edtFound;
//    if(!(edtFound = artsDequePopFront(artsThreadInfo.myNodeDeque)))
//    {
//        if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
//        {
//            if(!(edtFound = artsRuntimeStealFromNetwork()))
//                edtFound = artsRuntimeStealFromWorker();
//        }
//    }
//
//    if(edtFound)
//    {
//        artsRunEdt(edtFound);
//        return true;
//    }
    return false;
}

bool artsDefaultSchedulerLoop()
{
    struct artsEdt * edtFound = NULL;
    if(!(edtFound = artsDequePopFront(artsThreadInfo.myDeque)))
    {
//        if(!(edtFound = artsCheckLastWorker()))
        if(!(edtFound = artsRuntimeStealFromWorkerMult()))
        {
            artsUpdatePerformanceMetric(artsEdtStealAttempt, artsThread, 1, false);
    //        if(!(edtFound = (struct artsEdt*)dequeue(artsNodeInfo.numaQueue[artsThreadInfo.clusterId])))
            {
                if(!(edtFound = artsRuntimeStealFromWorker()))
                {
                    edtFound = artsRuntimeStealFromNetwork();
                }
                else
                {
                    artsUpdatePerformanceMetric(artsEdtSteal, artsThread, 1, false);
                }
            }
        }
    }

    if(edtFound)
    {
        artsRunEdt(edtFound);
        return true;
    }
    else
    {
        usleep(1);
    }
    return false;
}

static inline bool lockNetwork( volatile unsigned int * lock)
{

    if(*lock == 0U)
    {
        if(artsAtomicCswap( lock, 0U, artsThreadInfo.threadId+1U ) == 0U)
            return true;
    }

    return false;
}

static inline void unlockNetwork( volatile unsigned int * lock)
{
    *lock=0U;
}

int artsRuntimeLoop()
{
    ARTSCOUNTERTIMERSTART(totalCounter);
    if(artsThreadInfo.networkReceive)
    {
        while(artsThreadInfo.alive)
        {
            artsServerTryToRecieve(&artsNodeInfo.buf, &artsNodeInfo.packetSize, &artsNodeInfo.stealRequestLock);
        }
    }
    else if(artsThreadInfo.networkSend)
    {
        while(artsThreadInfo.alive)
        {
            if(artsNodeInfo.shutdownStarted && artsNodeInfo.shutdownTimeout > artsGetTimeStamp())
                artsRuntimeStop();
            else
                artsRemoteAsyncSend();
        }
    }
    else if(artsThreadInfo.worker)
    {
        while(artsThreadInfo.alive)
        {
            artsNodeInfo.scheduler();
        }
    }
    ARTSCOUNTERTIMERENDINCREMENT(totalCounter);
}
