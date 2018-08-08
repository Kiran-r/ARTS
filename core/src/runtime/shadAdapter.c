#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveGlobals.h"
#include "hiveAtomics.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveRuntime.h"
#include "hiveEdtFunctions.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveDebug.h"
#include "hiveTerminationDetection.h"
#include "hiveArrayList.h"
#include "hiveQueue.h"
#include <stdarg.h>
#include <string.h>

hiveGuid_t hiveEdtCreateShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + sizeof(hiveGuid_t);
    hiveGuid_t guid = NULL_GUID;
    hiveEdtCreateInternal(&guid, route, hiveThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 0, false, NULL_GUID, true);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

hiveGuid_t hiveActiveMessageShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, void * data, unsigned int size, hiveGuid_t epochGuid)
{
    unsigned int rank = route; //route / numNumaDomains;
    unsigned int cluster = 0; //route % numNumaDomains;
    hiveGuid_t guid = NULL_GUID;
    bool useEpoch = (epochGuid != NULL_GUID);
    
//    if(route == hiveGlobalRankId && hiveInspecting())
//    {
//        uint64_t edts = hiveGetPerformanceMetricTotal(hiveEdtThroughput, hiveNode);
//        uint64_t queued = hiveGetPerformanceMetricTotal(hiveEdtQueue, hiveNode);
//        uint64_t totalQueued = (queued > edts) ? queued - edts : 0;
//        double queueRate = hiveMetricTest(hiveEdtThroughput, hiveNode, totalQueued);
//        double doneRate = hiveMetricTest(hiveEdtThroughput, hiveNode, 1);
//        
//        
////        PRINTF("doneRate: %lf queueRate: %lf edts: %lu queued: %lu\n", doneRate, queueRate, edts, queued);
////        if(queueRate != 0 && doneRate != 0)
//        {
//            if( hiveNodeInfo.workerThreadCount==1 ||
//                ( !edts && hiveNodeInfo.workerThreadCount * 2 > queued) ||
//                ( totalQueued && (queueRate > doneRate) )
//              )
//            {
////                PRINTF("%lf * 1.5 = %lf < %lf Queued: %lu\n", doneRate, doneRate*1.5, queueRate, totalQueued);
//                hiveEdtDep_t dep;
//                dep.ptr = data;
//                dep.mode = DB_MODE_PTR;
//                dep.guid = NULL_GUID;
//                
//                HIVECOUNTERTIMERSTART(edtCounter);
//                
//                hiveGuid_t result = funcPtr(paramc, paramv, 1, &dep);
//                
//                HIVECOUNTERTIMERENDINCREMENT(edtCounter);
//                hiveUpdatePerformanceMetric(hiveEdtThroughput, hiveThread, 1, false);
//                
//                return NULL_GUID;
//            }
//        }
//    }
    
    if(size) {
        unsigned int depSpace = sizeof(hiveEdtDep_t);
        unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depSpace;
        hiveEdtCreateInternal(&guid, rank, cluster, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 1, useEpoch, epochGuid, true);
        
//        PRINTF("MEMCPY: %u\n", size);
        void * ptr = hiveMalloc(size);
        memcpy(ptr, data, size);
        hiveSignalEdtPtr(guid, NULL_GUID, ptr, size, 0);
    }
    else
    {
        unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64);
        hiveEdtCreateInternal(&guid, rank, cluster, edtSpace, NULL_GUID, funcPtr, paramc, paramv, 0, useEpoch, epochGuid, false);
    }
    return guid;
}

void hiveSynchronousActiveMessageShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, void * data, unsigned int size)
{
    unsigned int rank = route; //route / numNumaDomains;
    unsigned int cluster = 0; //route % numNumaDomains;
    unsigned int waitFlag = 1;
    void * waitPtr = &waitFlag;
    hiveGuid_t waitGuid = hiveAllocateLocalBuffer((void **)&waitPtr, sizeof(unsigned int), 1, NULL_GUID);
    
    hiveGuid_t guid = NULL_GUID;
    if(size) {
        unsigned int depSpace = sizeof(hiveEdtDep_t);
        unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depSpace;
        hiveEdtCreateInternal(&guid, rank, cluster, edtSpace, waitGuid, funcPtr, paramc, paramv, 1, false, NULL_GUID, true);

        void * ptr = hiveMalloc(size);
        memcpy(ptr, data, size);
        hiveSignalEdtPtr(guid, NULL_GUID, ptr, size, 0);
    }
    else
    {
        unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64);
        hiveEdtCreateInternal(&guid, rank, cluster, edtSpace, waitGuid, funcPtr, paramc, paramv, 0, false, NULL_GUID, false);
    }
    
    while(waitFlag) {
        hiveYield();
    }
}

void hiveIncLockShad()
{
    hiveThreadInfo.shadLock++;
}

void hiveDecLockShad()
{
    hiveThreadInfo.shadLock--;
}

void hiveCheckLockShad()
{
    if(hiveThreadInfo.shadLock)
    {
        PRINTF("ARTS: Cannot perform synchronous call under lock Worker: %u ShadLock: %u\n", hiveThreadInfo.groupId, hiveThreadInfo.shadLock);
        hiveDebugGenerateSegFault();
    }
}

void hiveStartIntroShad(unsigned int start)
{
    hiveStartInspector(start);
    HIVESTARTCOUNTING(start);
}

void hiveStopIntroShad()
{
    hiveStopInspector();
    HIVECOUNTERSOFF();
}

unsigned int hiveGetShadLoopStride()
{
    return hiveNodeInfo.shadLoopStride;
}

hiveGuid_t hiveAllocateLocalBufferShad(void ** buffer, uint32_t * sizeToWrite, hiveGuid_t epochGuid)
{
    if(epochGuid)
        incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    
    hiveBuffer_t * stub = hiveMalloc(sizeof(hiveBuffer_t));
    stub->buffer = *buffer;
    stub->sizeToWrite = sizeToWrite;
    stub->size = 0;
    stub->uses = 1;
    stub->epochGuid = epochGuid;
    
    hiveGuid_t guid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_BUFFER);
    hiveRouteTableAddItem(stub, guid, hiveGlobalRankId, false);
    return guid;
}

