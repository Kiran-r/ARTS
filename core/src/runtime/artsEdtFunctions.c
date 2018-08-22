#include "arts.h"
#include "artsMalloc.h"
#include "artsGuid.h"
#include "artsRemote.h"
#include "artsRemoteFunctions.h"
#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsCounter.h"
#include "artsIntrospection.h"
#include "artsRuntime.h"
#include "artsEdtFunctions.h"
#include "artsEventFunctions.h"
#include "artsOutOfOrder.h"
#include "artsRouteTable.h"
#include "artsDebug.h"
#include "artsTerminationDetection.h"
#include "artsArrayList.h"
#include "artsQueue.h"
#include <stdarg.h>
#include <string.h>

#define DPRINTF( ... )

#define maxEpochArrayList 32

extern unsigned int numNumaDomains;

__thread artsArrayList * epochList = NULL;
__thread struct artsEdt * currentEdt = NULL;

bool artsSetCurrentEpochGuid(artsGuid_t epochGuid)
{
    if(epochGuid)
    {   
        if(!epochList)
            epochList = artsNewArrayList(sizeof(artsGuid_t), 8);
        artsPushToArrayList(epochList, &epochGuid);
        if(currentEdt)
        {
            currentEdt->epochGuid = epochGuid;
            return true;
        }
    }
    return false;
}

artsGuid_t artsGetCurrentEpochGuid()
{
    if(epochList)
    {
        uint64_t length = artsLengthArrayList(epochList);
        if(length)
        {
            artsGuid_t * guid = artsGetFromArrayList(epochList, length-1);
            return *guid;
        }
    }
    return NULL_GUID;
}

artsGuid_t * artsCheckEpochIsRoot(artsGuid_t toCheck)
{
    if(epochList)
    {
        uint64_t length = artsLengthArrayList(epochList);
        for(uint64_t i=0; i<length; i++)
        {
            artsGuid_t * guid = artsGetFromArrayList(epochList, i);
            if(*guid == toCheck)
                return guid;
        }
    }
    PRINTF("ERROR %lu is not a valid epoch\n", toCheck);
    return NULL;
}

void artsSetThreadLocalEdtInfo(struct artsEdt * edt)
{
    artsThreadInfo.currentEdtGuid = edt->currentEdt;
    currentEdt = edt;
    
    if(epochList)
        artsResetArrayList(epochList);
    
    artsSetCurrentEpochGuid(currentEdt->epochGuid);
}

void artsSaveThreadLocal(threadLocal_t * tl)
{
    if(currentEdt)
    {
        ARTSCOUNTERTIMERENDINCREMENTBY(edtCounter, 0);
    }
    
    ARTSCOUNTERTIMERSTART(contextSwitch);
    tl->currentEdtGuid = artsThreadInfo.currentEdtGuid;
    tl->currentEdt = currentEdt;
    tl->epochList = (void*)epochList;
    
    artsThreadInfo.currentEdtGuid = NULL_GUID;
    currentEdt = NULL;
    epochList = NULL;
    ARTSCOUNTERTIMERENDINCREMENTBY(contextSwitch, 0);
    artsUpdatePerformanceMetric(artsYieldBW, artsThread, 1, false);
}

void artsRestoreThreadLocal(threadLocal_t * tl)
{
    ARTSCOUNTERTIMERSTART(contextSwitch);
    artsThreadInfo.currentEdtGuid = tl->currentEdtGuid;
    currentEdt = tl->currentEdt;
    if(epochList)
        artsDeleteArrayList(epochList);
    epochList = tl->epochList;
    ARTSCOUNTERTIMERENDINCREMENT(contextSwitch);
    
    ARTSCOUNTERTIMERSTART(edtCounter);
}

void artsIncrementFinishedEpochList()
{
    if(epochList)
    {
        
        unsigned int epochArrayLength = artsLengthArrayList(epochList);
        for(unsigned int i=0; i<epochArrayLength; i++)
        {
            artsGuid_t * guid = artsGetFromArrayList(epochList, i);
            DPRINTF("%lu Unsetting guid: %lu\n", artsThreadInfo.currentEdtGuid, guid);
            if(*guid)
                incrementFinishedEpoch(*guid);
        }
        
        if(epochArrayLength > maxEpochArrayList)
        {
            artsDeleteArrayList(epochList);
            epochList = NULL;
        }
        else         
            artsResetArrayList(epochList);
    }
    globalShutdownGuidIncFinished();
}

void artsUnsetThreadLocalEdtInfo()
{
    artsIncrementFinishedEpochList();
    artsThreadInfo.currentEdtGuid = NULL_GUID;
    currentEdt = NULL;
}

bool artsEdtCreateInternal(artsGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, artsGuid_t eventGuid, artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, bool useEpoch, artsGuid_t epochGuid, bool hasDepv)
{
    struct artsEdt *edt;
    ARTSSETMEMSHOTTYPE(artsEdtMemorySize);
    edt = (struct artsEdt*)artsCalloc(edtSpace);
    edt->header.type = ARTS_EDT;
    edt->header.size = edtSpace;
    ARTSSETMEMSHOTTYPE(artsDefaultMemorySize);
    if(edt)
    {
        bool createdGuid = false;
        if(*guid == NULL_GUID)
        {
            createdGuid = true;
            *guid = artsGuidCreateForRank(route, ARTS_EDT);
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
            artsGuid_t currentEpochGuid = NULL_GUID;
            if(epochGuid && artsCheckEpochIsRoot(epochGuid))
                currentEpochGuid = epochGuid;
            else
                currentEpochGuid = artsGetCurrentEpochGuid();

            if(currentEpochGuid)
            {
                edt->epochGuid = currentEpochGuid;
                incrementActiveEpoch(currentEpochGuid);
            }
        }
        globalShutdownGuidIncActive();
        
        if(paramc)
            memcpy((u64*) (edt+1), paramv, sizeof(u64) * paramc);

        if(eventGuid != NULL_GUID && artsGuidGetType(eventGuid) == ARTS_EVENT)
            artsAddDependence(*guid, eventGuid, ARTS_EVENT_LATCH_DECR_SLOT);

        if(route != artsGlobalRankId)
            artsRemoteMemoryMove(route, *guid, (void*)edt, (unsigned int)edt->header.size, ARTS_REMOTE_EDT_MOVE_MSG, artsFree);
        else
        {
            if(createdGuid) //this is a brand new edt
            {
                artsRouteTableAddItem(edt, *guid, artsGlobalRankId, false);
                if(edt->depcNeeded == 0)
                    artsHandleReadyEdt((void*)edt);
            }
            else //we are racing to add an edt
            {
                artsRouteTableAddItemRace(edt, *guid, artsGlobalRankId, false);
                if(edt->depcNeeded)
                {
                    artsRouteTableFireOO(*guid, artsOutOfOrderHandler); //Check the OO callback for EDT
                }
                else
                    artsHandleReadyEdt((void*)edt);
            }
        }
        return true;
    }
    return false;
}

/*----------------------------------------------------------------------------*/

artsGuid_t artsEdtCreateDep(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv)
{
    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(u64) + depSpace;
    artsGuid_t guid = NULL_GUID;
    artsEdtCreateInternal(&guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

artsGuid_t artsEdtCreateWithGuidDep(artsEdt_t funcPtr, artsGuid_t guid, u32 paramc, u64 * paramv, u32 depc, bool hasDepv)
{
    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int route = artsGuidGetRank(guid);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(u64) + depSpace;
    bool ret = artsEdtCreateInternal(&guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

artsGuid_t artsEdtCreateWithEpochDep(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, artsGuid_t epochGuid, bool hasDepv)
{
    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(u64) + depSpace;
    artsGuid_t guid = NULL_GUID;
    artsEdtCreateInternal(&guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, epochGuid, hasDepv);
    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

/*----------------------------------------------------------------------------*/

artsGuid_t artsEdtCreate(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc)
{
    return artsEdtCreateDep(funcPtr, route, paramc, paramv, depc, true);
}

artsGuid_t artsEdtCreateWithGuid(artsEdt_t funcPtr, artsGuid_t guid, u32 paramc, u64 * paramv, u32 depc)
{
    return artsEdtCreateWithGuidDep(funcPtr, guid, paramc, paramv, depc, true);
}

artsGuid_t artsEdtCreateWithEpoch(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, artsGuid_t epochGuid)
{
    return artsEdtCreateWithEpochDep(funcPtr, route, paramc, paramv, depc, epochGuid, true);
}

/*----------------------------------------------------------------------------*/

void artsEdtFree(struct artsEdt * edt)
{
    artsThreadInfo.edtFree = 1;
    artsFree(edt);
    artsThreadInfo.edtFree = 0;
}

inline void artsEdtDelete(struct artsEdt * edt)
{
    artsRouteTableRemoveItem(edt->currentEdt);
    artsEdtFree(edt);
}

void artsEdtDestroy(artsGuid_t guid)
{
    struct artsEdt * edt = (struct artsEdt*)artsRouteTableLookupItem(guid);
    artsRouteTableRemoveItem(guid);
    artsEdtFree(edt);
}

extern const char * const _artsTypeName[];

void internalSignalEdt(artsGuid_t edtPacket, u32 slot, artsGuid_t dataGuid, artsType_t mode, void * ptr, unsigned int size)
{
    ARTSEDTCOUNTERTIMERSTART(signalEdtCounter);  
    //This is old CDAG code... 
    if(currentEdt && currentEdt->invalidateCount > 0)
    {
        if(mode == ARTS_PTR)
            artsOutOfOrderSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
        else
            artsOutOfOrderSignalEdt(currentEdt->currentEdt, edtPacket, dataGuid, slot, mode);
    }
    else
    {
        unsigned int rank = artsGuidGetRank(edtPacket);
        if(rank == artsGlobalRankId)
        {
            struct artsEdt * edt = artsRouteTableLookupItem(edtPacket);
            if(edt)
            {
                artsEdtDep_t *edtDep = (artsEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
                if(slot < edt->depc)
                {
                    edtDep[slot].guid = dataGuid;
                    edtDep[slot].mode = mode;
                    edtDep[slot].ptr = ptr;
                }
                unsigned int res = artsAtomicSub(&edt->depcNeeded, 1U);
                DPRINTF("SIG: %lu %lu %u %p %d res: %u %s\n", edtPacket, dataGuid, slot, ptr, mode, res, getTypeName(edtDep[slot].mode));
                if(res == 0)
                    artsHandleReadyEdt(edt);
            }
            else
            {
                if(mode == ARTS_PTR)
                    artsOutOfOrderSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
                else
                    artsOutOfOrderSignalEdt(edtPacket, edtPacket, dataGuid, slot, mode);
            }
        }
        else
        {
            if(mode == ARTS_PTR)
                artsRemoteSignalEdtWithPtr(edtPacket, dataGuid, ptr, size, slot);
            else
                artsRemoteSignalEdt(edtPacket, dataGuid, slot, mode);
        }
    }
    artsUpdatePerformanceMetric(artsEdtSignalThroughput, artsThread, 1, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(signalEdtCounter);
}

void artsSignalEdt(artsGuid_t edtGuid, u32 slot, artsGuid_t dataGuid)
{
    artsGuid_t acqGuid = dataGuid;
    artsType_t mode = artsGuidGetType(dataGuid);
    if(mode == ARTS_DB_WRITE)
    {
        acqGuid = artsGuidCast(dataGuid, ARTS_DB_READ);
    }
    internalSignalEdt(edtGuid, slot, acqGuid, mode, NULL, 0);
}

void artsSignalEdtValue(artsGuid_t edtGuid, u32 slot, u64 value)
{
    internalSignalEdt(edtGuid, slot, value, ARTS_SINGLE_VALUE, NULL, 0);
}

void artsSignalEdtPtr(artsGuid_t edtGuid,  u32 slot, void * ptr, unsigned int size)
{
    internalSignalEdt(edtGuid, slot, NULL_GUID, ARTS_PTR, ptr, size);
}

artsGuid_t artsActiveMessageWithDb(artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, artsGuid_t dbGuid)
{
    unsigned int rank = artsGuidGetRank(dbGuid);
    artsGuid_t guid = artsEdtCreate(funcPtr, rank, paramc, paramv, depc+1);
//    PRINTF("AM -> %lu rank: %u depc: %u\n", guid, rank, depc+1);
    artsSignalEdt(guid, 0, dbGuid);
    return guid;
}

artsGuid_t artsActiveMessageWithDbAt(artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, artsGuid_t dbGuid, unsigned int rank)
{
    artsGuid_t guid = artsEdtCreate(funcPtr, rank, paramc, paramv, depc+1);
    PRINTF("AM -> %lu rank: %u depc: %u\n", guid, rank, depc+1);
    artsSignalEdt(guid, 0, dbGuid);
    return guid;
}

artsGuid_t artsActiveMessageWithBuffer(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * data, unsigned int size)
{
    void * ptr = artsMalloc(size);
    memcpy(ptr, data, size);
    artsGuid_t guid = artsEdtCreate(funcPtr, route, paramc, paramv, depc+1);
    artsSignalEdtPtr(guid, 0, ptr, size);
    return guid;
}

/*----------------------------------------------------------------------------*/

artsGuid_t artsAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, artsGuid_t epochGuid)
{
    if(epochGuid)
        incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
      
    if(size)
    {
        if(*buffer == NULL)
            *buffer = artsMalloc(sizeof(char) * size);
    }
    
    artsBuffer_t * stub = artsMalloc(sizeof(artsBuffer_t));
    stub->buffer = *buffer;
    stub->sizeToWrite = NULL;
    stub->size = size;
    stub->uses = uses;
    stub->epochGuid = epochGuid;
    
    artsGuid_t guid = artsGuidCreateForRank(artsGlobalRankId, ARTS_BUFFER);
    artsRouteTableAddItem(stub, guid, artsGlobalRankId, false);
    return guid;
}

void * artsSetBuffer(artsGuid_t bufferGuid, void * buffer, unsigned int size)
{
    void * ret = NULL;
    unsigned int rank = artsGuidGetRank(bufferGuid);
    if(rank==artsGlobalRankId)
    {
        artsBuffer_t * stub = artsRouteTableLookupItem(bufferGuid);
        if(stub)
        {
            artsGuid_t epochGuid = stub->epochGuid;
            if(epochGuid)
                incrementQueueEpoch(epochGuid);
            globalShutdownGuidIncQueue();
            
            if(size > stub->size)
            {
                if(stub->size)
                {
                    PRINTF("Truncating buffer data buffer size: %u stub size: %u\n", size, stub->size);
                    artsDebugPrintStack();
                }
                else if(stub->buffer == NULL)
                {
                    stub->buffer = artsMalloc(sizeof(char) * size);
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
            if(!artsAtomicSub(&stub->uses, 1))
            {
                artsRouteTableRemoveItem(bufferGuid);
                artsFree(stub);
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
//        artsRemoteMemoryMoveNoFree(rank, bufferGuid, buffer, size, ARTS_REMOTE_BUFFER_SEND_MSG);
        artsRemoteMemoryMove(rank, bufferGuid, buffer, size, ARTS_REMOTE_BUFFER_SEND_MSG, artsFree);
    }
    return ret;
}

void * artsGetBuffer(artsGuid_t bufferGuid)
{
    void * buffer = NULL;
    if(artsIsGuidLocal(bufferGuid))
    {
        artsBuffer_t * stub = artsRouteTableLookupItem(bufferGuid);
        buffer = stub->buffer;
        if(!artsAtomicSub(&stub->uses, 1))
        {
            artsRouteTableRemoveItem(bufferGuid);
            artsFree(stub);
        }
    }
    return buffer;
}
