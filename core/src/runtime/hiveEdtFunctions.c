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
#include "hiveEventFunctions.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveDebug.h"
#include "hiveTerminationDetection.h"
#include "hiveArrayList.h"
#include "hiveQueue.h"
#include <stdarg.h>
#include <string.h>

#define DPRINTF( ... )

#define maxEpochArrayList 32

extern unsigned int numNumaDomains;

__thread hiveArrayList * epochList = NULL;
__thread struct hiveEdt * currentEdt = NULL;

bool hiveSetCurrentEpochGuid(hiveGuid_t epochGuid)
{
    if(epochGuid)
    {   
        if(!epochList)
            epochList = hiveNewArrayList(sizeof(hiveGuid_t), 8);
        hivePushToArrayList(epochList, &epochGuid);
        if(currentEdt)
        {
            currentEdt->epochGuid = epochGuid;
            return true;
        }
    }
    return false;
}

hiveGuid_t hiveGetCurrentEpochGuid()
{
    if(epochList)
    {
        uint64_t length = hiveLengthArrayList(epochList);
        if(length)
        {
            hiveGuid_t * guid = hiveGetFromArrayList(epochList, length-1);
            return *guid;
        }
    }
    return NULL_GUID;
}

hiveGuid_t * hiveCheckEpochIsRoot(hiveGuid_t toCheck)
{
    if(epochList)
    {
        uint64_t length = hiveLengthArrayList(epochList);
        for(uint64_t i=0; i<length; i++)
        {
            hiveGuid_t * guid = hiveGetFromArrayList(epochList, i);
            if(*guid == toCheck)
                return guid;
        }
    }
    PRINTF("ERROR %lu is not a valid epoch\n", toCheck);
    return NULL;
}

void hiveSetThreadLocalEdtInfo(struct hiveEdt * edt)
{
    hiveThreadInfo.currentEdtGuid = edt->currentEdt;
    currentEdt = edt;
    
    if(epochList)
        hiveResetArrayList(epochList);
    
    hiveSetCurrentEpochGuid(currentEdt->epochGuid);
}

void hiveSaveThreadLocal(threadLocal_t * tl)
{
    if(currentEdt)
    {
        HIVECOUNTERTIMERENDINCREMENTBY(edtCounter, 0);
    }
    
    HIVECOUNTERTIMERSTART(contextSwitch);
    tl->currentEdtGuid = hiveThreadInfo.currentEdtGuid;
    tl->currentEdt = currentEdt;
    tl->epochList = (void*)epochList;
    
    hiveThreadInfo.currentEdtGuid = NULL_GUID;
    currentEdt = NULL;
    epochList = NULL;
    HIVECOUNTERTIMERENDINCREMENTBY(contextSwitch, 0);
    hiveUpdatePerformanceMetric(hiveYieldBW, hiveThread, 1, false);
}

void hiveRestoreThreadLocal(threadLocal_t * tl)
{
    HIVECOUNTERTIMERSTART(contextSwitch);
    hiveThreadInfo.currentEdtGuid = tl->currentEdtGuid;
    currentEdt = tl->currentEdt;
    if(epochList)
        hiveDeleteArrayList(epochList);
    epochList = tl->epochList;
    HIVECOUNTERTIMERENDINCREMENT(contextSwitch);
    
    HIVECOUNTERTIMERSTART(edtCounter);
}

void hiveIncrementFinishedEpochList()
{
    if(epochList)
    {
        
        unsigned int epochArrayLength = hiveLengthArrayList(epochList);
        for(unsigned int i=0; i<epochArrayLength; i++)
        {
            hiveGuid_t * guid = hiveGetFromArrayList(epochList, i);
            DPRINTF("%lu Unsetting guid: %lu\n", hiveThreadInfo.currentEdtGuid, guid);
            if(*guid)
                incrementFinishedEpoch(*guid);
        }
        
        if(epochArrayLength > maxEpochArrayList)
        {
            hiveDeleteArrayList(epochList);
            epochList = NULL;
        }
        else         
            hiveResetArrayList(epochList);
    }
    globalShutdownGuidIncFinished();
}

void hiveUnsetThreadLocalEdtInfo()
{
    hiveIncrementFinishedEpochList();
    hiveThreadInfo.currentEdtGuid = NULL_GUID;
    currentEdt = NULL;
}

bool hiveEdtCreateInternal(hiveGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, hiveGuid_t eventGuid, hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, bool useEpoch, hiveGuid_t epochGuid, bool hasDepv)
{
    struct hiveEdt *edt;
    HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
    edt = (struct hiveEdt*)hiveCalloc(edtSpace);
    edt->header.type = HIVE_EDT;
    edt->header.size = edtSpace;
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    if(edt)
    {
        bool createdGuid = false;
        if(*guid == NULL_GUID)
        {
            createdGuid = true;
            *guid = hiveGuidCreateForRank(route, HIVE_EDT);
        }
        
        edt->funcPtr = funcPtr;
        edt->depc = (hasDepv) ? depc : 0;
        edt->paramc = paramc;
        edt->currentEdt = *guid;
        edt->outputEvent = NULL_GUID;
        edt->epochGuid = NULL_GUID;
        edt->cluster = cluster;
        edt->depcNeeded = depc;
        edt->outputEvent = eventGuid;

        if(useEpoch)
        {
            hiveGuid_t currentEpochGuid = NULL_GUID;
            if(epochGuid && hiveCheckEpochIsRoot(epochGuid))
                currentEpochGuid = epochGuid;
            else
                currentEpochGuid = hiveGetCurrentEpochGuid();

            if(currentEpochGuid)
            {
                edt->epochGuid = currentEpochGuid;
                incrementActiveEpoch(currentEpochGuid);
            }
        }
        globalShutdownGuidIncActive();
        
        if(paramc)
            memcpy((u64*) (edt+1), paramv, sizeof(u64) * paramc);

        if(eventGuid != NULL_GUID && hiveGuidGetType(eventGuid) == HIVE_EVENT)
            hiveAddDependence(*guid, eventGuid, HIVE_EVENT_LATCH_DECR_SLOT);

        if(route != hiveGlobalRankId)
            hiveRemoteMemoryMove(route, *guid, (void*)edt, (unsigned int)edt->header.size, HIVE_REMOTE_EDT_MOVE_MSG, hiveFree);
        else
        {
            if(createdGuid) //this is a brand new edt
            {
                hiveRouteTableAddItem(edt, *guid, hiveGlobalRankId, false);
                if(edt->depcNeeded == 0)
                    hiveHandleReadyEdt((void*)edt);
            }
            else //we are racing to add an edt
            {
                hiveRouteTableAddItemRace(edt, *guid, hiveGlobalRankId, false);
                if(edt->depcNeeded)
                {
                    hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler); //Check the OO callback for EDT
                }
                else
                    hiveHandleReadyEdt((void*)edt);
            }
        }
        return true;
    }
    return false;
}

/*----------------------------------------------------------------------------*/

hiveGuid_t hiveEdtCreateDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(hiveEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depSpace;
    hiveGuid_t guid = NULL_GUID;
    hiveEdtCreateInternal(&guid, route, hiveThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

hiveGuid_t hiveEdtCreateWithGuidDep(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc, bool hasDepv)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int route = hiveGuidGetRank(guid);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(hiveEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depSpace;
    bool ret = hiveEdtCreateInternal(&guid, route, hiveThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

hiveGuid_t hiveEdtCreateWithEpochDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid, bool hasDepv)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(hiveEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depSpace;
    hiveGuid_t guid = NULL_GUID;
    hiveEdtCreateInternal(&guid, route, hiveThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, epochGuid, hasDepv);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

/*----------------------------------------------------------------------------*/

hiveGuid_t hiveEdtCreate(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc)
{
    return hiveEdtCreateDep(funcPtr, route, paramc, paramv, depc, true);
}

hiveGuid_t hiveEdtCreateWithGuid(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc)
{
    return hiveEdtCreateWithGuidDep(funcPtr, guid, paramc, paramv, depc, true);
}

hiveGuid_t hiveEdtCreateWithEpoch(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid)
{
    return hiveEdtCreateWithEpochDep(funcPtr, route, paramc, paramv, depc, epochGuid, true);
}

/*----------------------------------------------------------------------------*/

void hiveEdtFree(struct hiveEdt * edt)
{
    hiveThreadInfo.edtFree = 1;
    hiveFree(edt);
    hiveThreadInfo.edtFree = 0;
}

inline void hiveEdtDelete(struct hiveEdt * edt)
{
    hiveRouteTableRemoveItem(edt->currentEdt);
    hiveEdtFree(edt);
}

void hiveEdtDestroy(hiveGuid_t guid)
{
    struct hiveEdt * edt = (struct hiveEdt*)hiveRouteTableLookupItem(guid);
    hiveRouteTableRemoveItem(guid);
    hiveEdtFree(edt);
}

void internalSignalEdt(hiveGuid_t edtPacket, u32 slot, hiveGuid_t dataGuid, hiveDbAccessMode_t mode, void * ptr, unsigned int size)
{
    HIVEEDTCOUNTERTIMERSTART(signalEdtCounter);    
    if(currentEdt && currentEdt->invalidateCount > 0)
    {
        if(DB_MODE_PTR)
            hiveOutOfOrderSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
        else
            hiveOutOfOrderSignalEdt(currentEdt->currentEdt, edtPacket, dataGuid, slot, mode);
    }
    else
    {
        struct hiveEdt * edt = hiveRouteTableLookupItem(edtPacket);
        if(edt)
        {
            hiveEdtDep_t *edtDep = (hiveEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
            if(slot < edt->depc)
            {
                edtDep[slot].guid = dataGuid;
                edtDep[slot].mode = mode;
                edtDep[slot].ptr = ptr;
            }
            unsigned int res = hiveAtomicSub(&edt->depcNeeded, 1U);
            DPRINTF("SIG: %lu %lu %u res: %u\n", edtPacket, dataGuid, slot, res);
            if(res == 0)
                hiveHandleReadyEdt(edt);
        }
        else
        {
            unsigned int rank = hiveGuidGetRank(edtPacket);
            if(rank != hiveGlobalRankId)
            {
                if(DB_MODE_PTR)
                    hiveOutOfOrderSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
                else
                    hiveRemoteSignalEdt(edtPacket, dataGuid, slot, mode);
            }
            else
            {
                rank = hiveRouteTableLookupRank(edtPacket);
                if(rank == hiveGlobalRankId || rank == -1)
                {
                    if(DB_MODE_PTR)
                        hiveOutOfOrderSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
                    else
                        hiveOutOfOrderSignalEdt(edtPacket, edtPacket, dataGuid, slot, mode);
                }
                else
                {
                    if(DB_MODE_PTR)
                        hiveRemoteSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
                    else
                        hiveRemoteSignalEdt(edtPacket, dataGuid, slot, mode);
                }
            }
        }
    }
    hiveUpdatePerformanceMetric(hiveEdtSignalThroughput, hiveThread, 1, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(signalEdtCounter);
}

void hiveSignalEdt(hiveGuid_t edtGuid, u32 slot, hiveGuid_t dataGuid)
{
    internalSignalEdt(edtGuid, dataGuid, slot, DB_MODE_NONE, NULL, 0);
}

void hiveSignalEdtValue(hiveGuid_t edtGuid, u32 slot, u64 value)
{
    internalSignalEdt(edtGuid, value, slot, DB_MODE_SINGLE_VALUE, NULL, 0);
}

void hiveSignalEdtPtr(hiveGuid_t edtGuid,  u32 slot, void * ptr, unsigned int size)
{
    internalSignalEdt(edtGuid, NULL_GUID, slot, DB_MODE_PTR, ptr, size);
}

hiveGuid_t hiveActiveMessageWithDb(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid)
{
    unsigned int rank = hiveGuidGetRank(dbGuid);
    hiveGuid_t guid = hiveEdtCreate(funcPtr, rank, paramc, paramv, depc+1);
//    PRINTF("AM -> %lu rank: %u depc: %u\n", guid, rank, depc+1);
    hiveSignalEdt(guid, 0, dbGuid);
    return guid;
}

hiveGuid_t hiveActiveMessageWithDbAt(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid, unsigned int rank)
{
    hiveGuid_t guid = hiveEdtCreate(funcPtr, rank, paramc, paramv, depc+1);
//    PRINTF("AM -> %lu rank: %u depc: %u\n", guid, rank, depc+1);
    hiveSignalEdt(guid, 0, dbGuid);
    return guid;
}

hiveGuid_t hiveActiveMessageWithBuffer(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * data, unsigned int size)
{
    void * ptr = hiveMalloc(size);
    memcpy(ptr, data, size);
    hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc, paramv, depc+1);
    hiveSignalEdtPtr(guid, 0, ptr, size);
    return guid;
}

/*----------------------------------------------------------------------------*/

hiveGuid_t hiveAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, hiveGuid_t epochGuid)
{
    if(epochGuid)
        incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
      
    if(size)
    {
        if(*buffer == NULL)
            *buffer = hiveMalloc(sizeof(char) * size);
    }
    
    hiveBuffer_t * stub = hiveMalloc(sizeof(hiveBuffer_t));
    stub->buffer = *buffer;
    stub->sizeToWrite = NULL;
    stub->size = size;
    stub->uses = uses;
    stub->epochGuid = epochGuid;
    
    hiveGuid_t guid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_BUFFER);
    hiveRouteTableAddItem(stub, guid, hiveGlobalRankId, false);
    return guid;
}

void * hiveSetBuffer(hiveGuid_t bufferGuid, void * buffer, unsigned int size)
{
    void * ret = NULL;
    unsigned int rank = hiveGuidGetRank(bufferGuid);
    if(rank==hiveGlobalRankId)
    {
        hiveBuffer_t * stub = hiveRouteTableLookupItem(bufferGuid);
        if(stub)
        {
            hiveGuid_t epochGuid = stub->epochGuid;
            if(epochGuid)
                incrementQueueEpoch(epochGuid);
            globalShutdownGuidIncQueue();
            
            if(size > stub->size)
            {
                if(stub->size)
                {
                    PRINTF("Truncating buffer data buffer size: %u stub size: %u\n", size, stub->size);
                    hiveDebugPrintStack();
                }
                else if(stub->buffer == NULL)
                {
                    stub->buffer = hiveMalloc(sizeof(char) * size);
                    stub->size = size;
                }
                else
                    stub->size = size;
            }
            
            if(stub->sizeToWrite)
                *stub->sizeToWrite = (uint32_t)size;
            
            memcpy(stub->buffer, buffer, stub->size);
//            PRINTF("Set buffer %p %u %u\n", stub->buffer, *((unsigned int*)stub->buffer), stub->size);
            ret = stub->buffer;
            if(!hiveAtomicSub(&stub->uses, 1))
            {
                hiveRouteTableRemoveItem(bufferGuid);
                hiveFree(stub);
            }
            
            if(epochGuid)
                incrementFinishedEpoch(epochGuid);
            globalShutdownGuidIncFinished();
        }
        else
            PRINTF("Out-of-order buffers not supported\n");
    }
    else
    {
//        PRINTF("Sending size: %u\n", size);
//        hiveRemoteMemoryMoveNoFree(rank, bufferGuid, buffer, size, HIVE_REMOTE_BUFFER_SEND_MSG);
        hiveRemoteMemoryMove(rank, bufferGuid, buffer, size, HIVE_REMOTE_BUFFER_SEND_MSG, hiveFree);
    }
    return ret;
}

void * hiveGetBuffer(hiveGuid_t bufferGuid)
{
    void * buffer = NULL;
    if(hiveIsGuidLocal(bufferGuid))
    {
        hiveBuffer_t * stub = hiveRouteTableLookupItem(bufferGuid);
        buffer = stub->buffer;
        if(!hiveAtomicSub(&stub->uses, 1))
        {
            hiveRouteTableRemoveItem(bufferGuid);
            hiveFree(stub);
        }
    }
    return buffer;
}
