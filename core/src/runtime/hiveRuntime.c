#include <unistd.h>
#include "hive.h"
#include "hiveMalloc.h"
#include "hiveDeque.h"
#include "hiveAtomics.h"
#include "hiveGuid.h"
#include "hiveCounter.h"
#include "hiveGlobals.h"
#include "hiveRemote.h"
#include "hiveOutOfOrder.h"
#include "hiveRuntime.h"
#include <inttypes.h>
#include "hiveCounter.h"
#include "hiveDebug.h"
#include "hiveRemoteFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveAbstractMachineModel.h"
#include "hiveRouteTable.h"
#include "hiveEdtFunctions.h"
#include "hiveEventFunctions.h"
#include "hiveThreads.h"
#include "hiveOutOfOrder.h"
#include "hiveIntrospection.h"
#include "hiveTerminationDetection.h"
#include <stdlib.h>
#include <stdio.h>
#include "hiveTimer.h"
#include "hiveArrayList.h"
#include "hiveQueue.h"

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

struct hiveRuntimeShared hiveNodeInfo;
__thread struct hiveRuntimePrivate hiveThreadInfo;

typedef bool (*scheduler_t)();
scheduler_t schedulerLoop[3] = {hiveDefaultSchedulerLoop, hiveNetworkBeforeStealSchedulerLoop, hiveNetworkFirstSchedulerLoop};

hiveGuid_t artsMainEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    artsMain(mainArgc, mainArgv);
    return NULL_GUID;
}

void hiveRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct hiveConfig * config)
{
    hiveThreadSetOsThreadCount(config->osThreadCount);
    hiveNodeInfo.scheduler = schedulerLoop[config->scheduler];
    hiveNodeInfo.deque = (struct hiveDeque**) hiveMalloc(sizeof(struct hiveDeque*)*totalThreads);
    hiveNodeInfo.receiverDeque = (struct hiveDeque**) hiveMalloc(sizeof(struct hiveDeque*)*receiverThreads);
    hiveNodeInfo.numaQueue = (struct hiveQueue**) hiveMalloc(sizeof(struct hiveQueue*)*numNumaDomains);
    for(unsigned int i=0; i<numNumaDomains; i++)
        hiveNodeInfo.numaQueue[i] = hiveNewQueue();
    hiveNodeInfo.routeTable = (struct hiveRouteTable**) hiveCalloc(sizeof(struct hiveRouteTable*)*totalThreads);
    hiveNodeInfo.remoteRouteTable = hiveRouteTableListNew(1, config->routeTableEntries, config->routeTableSize);
    hiveNodeInfo.localSpin = (volatile bool**) hiveCalloc(sizeof(bool*)*totalThreads);
    hiveNodeInfo.memoryMoves = (unsigned int**) hiveCalloc(sizeof(unsigned int*)*totalThreads);
    hiveNodeInfo.atomicWaits = (struct atomicCreateBarrierInfo **) hiveCalloc(sizeof(struct atomicCreateBarrierInfo*)*totalThreads);
    hiveNodeInfo.workerThreadCount = workerThreads;
    hiveNodeInfo.senderThreadCount = senderThreads;
    hiveNodeInfo.receiverThreadCount = receiverThreads;
    hiveNodeInfo.totalThreadCount       = totalThreads;
    hiveNodeInfo.readyToPush            = totalThreads;
    hiveNodeInfo.readyToParallelStart   = totalThreads;
    hiveNodeInfo.readyToInspect         = totalThreads;
    hiveNodeInfo.readyToExecute         = totalThreads;
    hiveNodeInfo.readyToClean           = totalThreads;
    hiveNodeInfo.sendLock = 0U;
    hiveNodeInfo.recvLock = 0U;
    hiveNodeInfo.shutdownCount = hiveGlobalRankCount-1;
    hiveNodeInfo.shutdownStarted=0;
    hiveNodeInfo.readyToShutdown = hiveGlobalRankCount-1;
    hiveNodeInfo.stealRequestLock = !remoteStealingOn;
    hiveNodeInfo.buf = hiveMalloc( PACKET_SIZE );
    hiveNodeInfo.packetSize = PACKET_SIZE;
    hiveNodeInfo.printNodeStats = config->printNodeStats;
    hiveNodeInfo.shutdownEpoch = (config->shutdownEpoch) ? 1 : NULL_GUID ;
    hiveNodeInfo.shadLoopStride = config->shadLoopStride;
    hiveInitIntrospector(config);
}

void hiveRuntimeGlobalCleanup()
{
    hiveIntrospectivePrintTotals(hiveGlobalRankId);
    hiveFree(hiveNodeInfo.deque);
    hiveFree((void *)hiveNodeInfo.localSpin);
    hiveFree(hiveNodeInfo.memoryMoves);
    hiveFree(hiveNodeInfo.atomicWaits);
}

void hiveThreadZeroNodeStart()
{
    hiveStartInspector(1);

    setGlobalGuidOn();
    createShutdownEpoch();
    
    if(initPerNode)
        initPerNode(hiveGlobalRankId, mainArgc, mainArgv);
    setGuidGeneratorAfterParallelStart();

    hiveStartInspector(2);
    HIVESTARTCOUNTING(2);
    hiveAtomicSub(&hiveNodeInfo.readyToParallelStart, 1U);
    while(hiveNodeInfo.readyToParallelStart){ }
    if(initPerWorker && hiveThreadInfo.worker)
        initPerWorker(hiveGlobalRankId, hiveThreadInfo.groupId, mainArgc, mainArgv);
    
    if(artsMain && !hiveGlobalRankId)
        hiveEdtCreate(artsMainEdt, 0, 0, NULL, 0);
    
    hiveIncrementFinishedEpochList();
    
    hiveAtomicSub(&hiveNodeInfo.readyToInspect, 1U);
    while(hiveNodeInfo.readyToInspect){ }
    HIVESTARTCOUNTING(3);
    hiveStartInspector(3);
    hiveAtomicSub(&hiveNodeInfo.readyToExecute, 1U);
    while(hiveNodeInfo.readyToExecute){ }
}

void hiveRuntimePrivateInit(struct threadMask * unit, struct hiveConfig  * config)
{
    hiveNodeInfo.deque[unit->id] = hiveThreadInfo.myDeque = hiveDequeNew(config->dequeSize);
    if(unit->worker)
    {
        hiveNodeInfo.routeTable[unit->id] =  hiveRouteTableListNew(1, config->routeTableEntries, config->routeTableSize);
    }

    if(unit->networkSend || unit->networkReceive)
    {
        if(unit->networkSend)
        {
            unsigned int size = hiveGlobalRankCount*config->ports / hiveNodeInfo.senderThreadCount;
            unsigned int rem = hiveGlobalRankCount*config->ports % hiveNodeInfo.senderThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                hiveRemotSetThreadOutboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                hiveRemotSetThreadOutboundQueues(start, start+size);
            }
        }
        if(unit->networkReceive)
        {
            hiveNodeInfo.receiverDeque[unit->groupPos] = hiveNodeInfo.deque[unit->id];
            unsigned int size = (hiveGlobalRankCount-1)*config->ports / hiveNodeInfo.receiverThreadCount;
            unsigned int rem = (hiveGlobalRankCount-1)*config->ports % hiveNodeInfo.receiverThreadCount;
            unsigned int start;
            if(unit->groupPos < rem)
            {
                start = unit->groupPos*(size+1);
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                hiveRemotSetThreadInboundQueues(start, start+size+1);
            }
            else
            {
                start = rem*(size+1) + (unit->groupPos - rem) * size ;
                //PRINTF("%d %d %d %d\n", start, size, unit->groupPos, rem);
                hiveRemotSetThreadInboundQueues(start, start+size);
            }

        }
    }
    hiveNodeInfo.localSpin[unit->id] = &hiveThreadInfo.alive;
    hiveThreadInfo.alive = true;
    hiveNodeInfo.memoryMoves[unit->id] = (unsigned int *)&hiveThreadInfo.oustandingMemoryMoves;
    hiveNodeInfo.atomicWaits[unit->id] = &hiveThreadInfo.atomicWait;
    hiveThreadInfo.atomicWait.wait = true;
    hiveThreadInfo.oustandingMemoryMoves = 0;
    hiveThreadInfo.coreId = unit->unitId;
    hiveThreadInfo.threadId = unit->id;
    hiveThreadInfo.groupId = unit->groupPos;
    hiveThreadInfo.clusterId = unit->clusterId;
    hiveThreadInfo.worker = unit->worker;
    hiveThreadInfo.networkSend = unit->networkSend;
    hiveThreadInfo.networkReceive = unit->networkReceive;
    hiveThreadInfo.backOff = 1;
    hiveThreadInfo.currentEdtGuid = 0;
    hiveThreadInfo.mallocType = hiveDefaultMemorySize;
    hiveThreadInfo.mallocTrace = 1;
    hiveThreadInfo.localCounting = 1;
    hiveThreadInfo.shadLock = 0;
    hiveGuidKeyGeneratorInit();
    INITCOUNTERLIST(unit->id, hiveGlobalRankId, config->counterFolder, config->counterStartPoint);
    hiveAtomicSub(&hiveNodeInfo.readyToPush, 1U);
    while(hiveNodeInfo.readyToPush){  };
    if(unit->id)
    {
        hiveAtomicSub(&hiveNodeInfo.readyToParallelStart, 1U);
        while(hiveNodeInfo.readyToParallelStart){ };

        if(hiveThreadInfo.worker)
        {
            if(initPerWorker)
                initPerWorker(hiveGlobalRankId, hiveThreadInfo.groupId, mainArgc, mainArgv);
            hiveIncrementFinishedEpochList();
        }
        
        hiveAtomicSub(&hiveNodeInfo.readyToInspect, 1U);
        while(hiveNodeInfo.readyToInspect) { };
        hiveAtomicSub(&hiveNodeInfo.readyToExecute, 1U);
        while(hiveNodeInfo.readyToExecute) { };
    }
    hiveThreadInfo.drand_buf[0] = 1202107158 + unit->id * 1999;
    hiveThreadInfo.drand_buf[1] = 0;
    hiveThreadInfo.drand_buf[2] = 0;
}

void hiveRuntimePrivateCleanup()
{
    hiveAtomicSub(&hiveNodeInfo.readyToClean, 1U);
    while(hiveNodeInfo.readyToClean){ };
    if(hiveThreadInfo.myDeque)
        hiveDequeDelete(hiveThreadInfo.myDeque);
    if(hiveThreadInfo.myNodeDeque)
        hiveDequeDelete(hiveThreadInfo.myNodeDeque);
#if defined(COUNT) || defined(MODELCOUNT)
    hiveWriteCountersToFile(hiveThreadInfo.threadId, hiveGlobalRankId);
#endif
    hiveWriteMetricShotFile(hiveThreadInfo.threadId, hiveGlobalRankId);
}

void hiveRuntimeStop()
{
    unsigned int i;
    for(i=0; i<hiveNodeInfo.totalThreadCount; i++)
    {
        while(!hiveNodeInfo.localSpin[i]);
        (*hiveNodeInfo.localSpin[i]) = false;
    }
    hiveStopInspector();
}

void hiveHandleRemoteStolenEdt(struct hiveEdt *edt)
{
    DPRINTF("push stolen %d\n",hiveThreadInfo.coreId);
    incrementQueueEpoch(edt->epochGuid);
    globalShutdownGuidIncQueue();
    hiveDequePushFront(hiveThreadInfo.myDeque, edt, 0);
}

void hiveHandleReadyEdt(struct hiveEdt * edt)
{
    acquireDbs(edt);
    if(hiveAtomicSub(&edt->depcNeeded,1U) == 0)
    {
        incrementQueueEpoch(edt->epochGuid);
        globalShutdownGuidIncQueue();
//        if(edt->cluster == hiveThreadInfo.clusterId)
            hiveDequePushFront(hiveThreadInfo.myDeque, edt, 0);
//        else
//            enqueue((uint64_t)edt, hiveNodeInfo.numaQueue[hiveThreadInfo.clusterId]);
//        enqueue((uint64_t)edt, hiveNodeInfo.numaQueue[edt->cluster]);
        hiveUpdatePerformanceMetric(hiveEdtQueue, hiveThread, 1, false);
    }
}

static inline void hiveRunEdt(void *edtPacket)
{
    struct hiveEdt *edt = edtPacket;
    u32 depc = edt->depc;
    hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);

    hiveEdt_t func = edt->funcPtr;
    u32 paramc = edt->paramc;
    u64 *paramv = (u64 *)(edt + 1);

    prepDbs(depc, depv);

    hiveSetThreadLocalEdtInfo(edt);
    HIVECOUNTERTIMERSTART(edtCounter);

    hiveGuid_t result = func(paramc, paramv, depc, depv);

    HIVECOUNTERTIMERENDINCREMENT(edtCounter);
    hiveUpdatePerformanceMetric(hiveEdtThroughput, hiveThread, 1, false);

    hiveUnsetThreadLocalEdtInfo();

    if(edt->outputEvent != NULL_GUID)
    {
        if(hiveGuidGetType(edt->outputEvent) == HIVE_EVENT)
            hiveEventSatisfySlot(edt->outputEvent, result, HIVE_EVENT_LATCH_DECR_SLOT);
        else //This is for a synchronous path
        {
            hiveSetBuffer(edt->outputEvent, hiveCalloc(sizeof(unsigned int)), sizeof(unsigned int));
        }
    }
    
    releaseDbs(depc, depv);
    hiveEdtDelete(edtPacket);
}

inline unsigned int hiveRuntimeStealAnyMultipleEdt( unsigned int amount, void ** returnList )
{
    struct hiveEdt *edt = NULL;
    unsigned int i;
    unsigned int count = 0;
    bool done = false;
    for (i=0; i<hiveNodeInfo.workerThreadCount && !done; i++)
    {
        do
        {
//            edt = hiveDequePopBack(hiveNodeInfo.workerDeque[i]);
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

inline struct hiveEdt * hiveRuntimeStealFromNetwork()
{
    struct hiveEdt *edt = NULL;
    if(hiveGlobalRankCount > 1)
    {
        unsigned int index = hiveThreadInfo.threadId;
        for (unsigned int i=0; i<hiveNodeInfo.receiverThreadCount; i++)
        {
            index = (index + 1) % hiveNodeInfo.receiverThreadCount;
            if(edt = hiveDequePopBack(hiveNodeInfo.receiverDeque[index]))
                break;
        }
    }
    return edt;
}

__thread unsigned int lastHitThread = 0;

inline struct hiveEdt * hiveCheckLastWorker() {
    struct hiveEdt * ret = hiveDequePopBack(hiveNodeInfo.deque[lastHitThread]);
    if(ret)
        hiveUpdatePerformanceMetric(hiveEdtLastLocalHit, hiveThread, 1, false);
    return ret;
}

inline struct hiveEdt * hiveRuntimeStealFromWorker()
{
    struct hiveEdt *edt = NULL;
    if(hiveNodeInfo.totalThreadCount > 1)
    {
        
        long unsigned int stealLoc;
        do
        {
            stealLoc = jrand48(hiveThreadInfo.drand_buf);
            stealLoc = stealLoc % hiveNodeInfo.totalThreadCount;
        } while(stealLoc == hiveThreadInfo.threadId);

        edt = hiveDequePopBack(hiveNodeInfo.deque[stealLoc]);
        
        if(edt)
        {
            lastHitThread = (unsigned int) stealLoc;
        }
    }
    return edt;
}

struct hiveEdt * hiveRuntimeStealFromWorkerMult()
{
    struct hiveEdt * edt = NULL;
    if(hiveNodeInfo.totalThreadCount > 1)
    {
        
        long unsigned int stealLoc;
        do
        {
            stealLoc = jrand48(hiveThreadInfo.drand_buf);
            stealLoc = stealLoc % hiveNodeInfo.totalThreadCount;
        } while(stealLoc == hiveThreadInfo.threadId);

        unsigned int size = hiveDequeSize(hiveNodeInfo.deque[stealLoc]);
        unsigned int half = size/2;
        unsigned int stolen = 0;
        if(STEALSIZE < half)
        {
            for(unsigned int i=0; i<half; i+=STEALSIZE) 
            {
                void ** array = hiveDequePopBackMult(hiveNodeInfo.deque[stealLoc]);
                if(array)
                {
                    for(unsigned int j=0; j<STEALSIZE; j++)
                    {
                        if(edt)
                            hiveDequePushFront(hiveThreadInfo.myDeque, array[j], 0);
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
                hiveUpdatePerformanceMetric(hiveEdtSteal, hiveThread, stolen, false);
                lastHitThread = (unsigned int) stealLoc;
            }
        }
    }
    return edt;
}

bool hiveNetworkFirstSchedulerLoop()
{
//    struct hiveEdt *edtFound;
//    if(!(edtFound = hiveRuntimeStealFromNetwork()))
//    {
//        if(!(edtFound = hiveDequePopFront(hiveThreadInfo.myNodeDeque)))
//        {
//            if(!(edtFound = hiveDequePopFront(hiveThreadInfo.myDeque)))
//                edtFound = hiveRuntimeStealFromWorker();
//        }
//    }
//    if(edtFound)
//    {
//        hiveRunEdt(edtFound);
//        return true;
//    }
    return false;
}

bool hiveNetworkBeforeStealSchedulerLoop()
{
//    struct hiveEdt *edtFound;
//    if(!(edtFound = hiveDequePopFront(hiveThreadInfo.myNodeDeque)))
//    {
//        if(!(edtFound = hiveDequePopFront(hiveThreadInfo.myDeque)))
//        {
//            if(!(edtFound = hiveRuntimeStealFromNetwork()))
//                edtFound = hiveRuntimeStealFromWorker();
//        }
//    }
//
//    if(edtFound)
//    {
//        hiveRunEdt(edtFound);
//        return true;
//    }
    return false;
}

bool hiveDefaultSchedulerLoop()
{
    struct hiveEdt * edtFound = NULL;
    if(!(edtFound = hiveDequePopFront(hiveThreadInfo.myDeque)))
    {
//        if(!(edtFound = hiveCheckLastWorker()))
        if(!(edtFound = hiveRuntimeStealFromWorkerMult()))
        {
            hiveUpdatePerformanceMetric(hiveEdtStealAttempt, hiveThread, 1, false);
    //        if(!(edtFound = (struct hiveEdt*)dequeue(hiveNodeInfo.numaQueue[hiveThreadInfo.clusterId])))
            {
                if(!(edtFound = hiveRuntimeStealFromWorker()))
                {
                    edtFound = hiveRuntimeStealFromNetwork();
                }
                else
                {
                    hiveUpdatePerformanceMetric(hiveEdtSteal, hiveThread, 1, false);
                }
            }
        }
    }

    if(edtFound)
    {
        hiveRunEdt(edtFound);
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
        if(hiveAtomicCswap( lock, 0U, hiveThreadInfo.threadId+1U ) == 0U)
            return true;
    }

    return false;
}

static inline void unlockNetwork( volatile unsigned int * lock)
{
    *lock=0U;
}

int hiveRuntimeLoop()
{
    HIVECOUNTERTIMERSTART(totalCounter);
    if(hiveThreadInfo.networkReceive)
    {
        while(hiveThreadInfo.alive)
        {
            hiveServerTryToRecieve(&hiveNodeInfo.buf, &hiveNodeInfo.packetSize, &hiveNodeInfo.stealRequestLock);
        }
    }
    else if(hiveThreadInfo.networkSend)
    {
        while(hiveThreadInfo.alive)
        {
            if(hiveNodeInfo.shutdownStarted && hiveNodeInfo.shutdownTimeout > hiveGetTimeStamp())
                hiveRuntimeStop();
            else
                hiveRemoteAsyncSend();
        }
    }
    else if(hiveThreadInfo.worker)
    {
        while(hiveThreadInfo.alive)
        {
            hiveNodeInfo.scheduler();
        }
    }
    HIVECOUNTERTIMERENDINCREMENT(totalCounter);
}
