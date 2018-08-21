#include "hive.h"
#include "hiveGuid.h"
#include "hiveHash.h"
#include "hiveRouteTable.h"
#include "hiveGlobals.h"
#include "hiveMalloc.h"
#include "hiveCounter.h"
#include "hiveDebug.h"
#include <stdint.h>
#include <string.h>

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

__thread u64 globalGuidThreadId = 0;
__thread u64 * keys;
u64 numTables = 0;
u64 keysPerThread = 0;
u64 globalGuidOn = 0;
u64 minGlobalGuidThread = 0;
u64 maxGlobalGuidThread = 0;

void setGlobalGuidOn()
{
    globalGuidOn = ((u64)1) << 40;
}

u64 * hiveGuidGeneratorGetKey(unsigned int route, unsigned int type )
{
    return &keys[route*HIVE_LAST_TYPE + type];
}

hiveGuid_t hiveGuidCreateForRankInternal(unsigned int route, unsigned int type, unsigned int guidCount)
{
    hiveGuid guid;
    if(globalGuidOn)
    {
        //Safeguard against wrap around
        if(globalGuidOn > guidCount)
        {
            guid.fields.key = globalGuidOn - guidCount;
            globalGuidOn -= guidCount;
        }
        else
        {
            PRINTF("Parallel Start out of guid keys\n");
            hiveDebugGenerateSegFault();
        }
    }
    else
    {
        
        u64 * key = hiveGuidGeneratorGetKey(route, type);
        u64 value = *key;
        if(value + guidCount < keysPerThread)
        {
            guid.fields.key = value + keysPerThread * globalGuidThreadId;
            (*key)+=guidCount;
        }
        else
        {
            PRINTF("Out of guid keys\n");
            hiveDebugGenerateSegFault();
        }
    }
    guid.fields.type = type;
    guid.fields.rank = route;
    DPRINTF("Key: %lu %lu %lu %p %lu sizeof(hiveGuid): %u parallel start: %u\n", guid.fields.type, guid.fields.rank, guid.fields.key, guid.bits, (hiveGuid_t) guid.bits, sizeof(hiveGuid), (globalGuidOn!=0));
    HIVEEDTCOUNTERINCREMENTBY(guidAllocCounter, guidCount);
    return (hiveGuid_t)guid.bits;
}

hiveGuid_t hiveGuidCreateForRank(unsigned int route, unsigned int type)
{
    return hiveGuidCreateForRankInternal(route, type, 1);
}

void setGuidGeneratorAfterParallelStart()
{
        unsigned int numOfTables = hiveNodeInfo.workerThreadCount + 1;
        keysPerThread = globalGuidOn / (numOfTables * hiveGlobalRankCount);
        DPRINTF("Keys per thread %lu\n", keysPerThread);
        globalGuidOn = 0;
}

void hiveGuidKeyGeneratorInit()
{
    numTables           = (hiveGlobalRankCount == 1) ? hiveNodeInfo.workerThreadCount : hiveNodeInfo.workerThreadCount + 1;
    u64 localId         = (hiveThreadInfo.worker) ? hiveThreadInfo.threadId : hiveNodeInfo.workerThreadCount;
    minGlobalGuidThread = numTables * hiveGlobalRankId;
    maxGlobalGuidThread = (hiveGlobalRankCount == 1) ? minGlobalGuidThread + numTables : minGlobalGuidThread + numTables -1;
    globalGuidThreadId  = minGlobalGuidThread + localId;
    
//    PRINTF("numTables: %lu localId: %lu minGlobalGuidThread: %lu maxGlobalGuidThread: %lu globalGuidThreadId: %lu\n", numTables, localId, minGlobalGuidThread, maxGlobalGuidThread, globalGuidThreadId);
    keys = hiveMalloc(sizeof(u64) * HIVE_LAST_TYPE * hiveGlobalRankCount);
    for(unsigned int i=0; i<HIVE_LAST_TYPE * hiveGlobalRankCount; i++)
        keys[i] = 1;
}

hiveGuid_t hiveGuidCast(hiveGuid_t guid, hiveType_t type)
{
    hiveGuid addressInfo = (hiveGuid) guid;
    addressInfo.fields.type = (unsigned int) type;
    return (hiveGuid_t) addressInfo.bits;
}

hiveType_t hiveGuidGetType(hiveGuid_t guid)
{
    hiveGuid addressInfo = (hiveGuid) guid;
    return addressInfo.fields.type;
}

unsigned int hiveGuidGetRank(hiveGuid_t guid)
{
    hiveGuid addressInfo = (hiveGuid) guid;
    return addressInfo.fields.rank;
}

bool hiveIsGuidLocal(hiveGuid_t guid)
{
    return (hiveGlobalRankId == hiveGuidGetRank(guid));
}

hiveGuid_t hiveReserveGuidRoute(hiveType_t type, unsigned int route)
{
    hiveGuid_t guid = NULL_GUID;
    route = route % hiveGlobalRankCount;
    if(type > HIVE_NULL && type < HIVE_LAST_TYPE)
        guid = hiveGuidCreateForRankInternal(route, (unsigned int)type, 1);
//    if(route == hiveGlobalRankId)
//        hiveRouteTableAddItem(NULL, guid, hiveGlobalRankId, false);    
    return guid;
}

hiveGuid_t * hiveReserveGuidsRoundRobin(unsigned int size, hiveType_t type)
{
    hiveGuid_t * guids = NULL;
    if(type > HIVE_NULL && type < HIVE_LAST_TYPE)
    {
        hiveGuid_t * guids = (hiveGuid_t*) hiveMalloc(size*sizeof(hiveGuid_t));
        for(unsigned int i=0; i<size; i++)
        {
            unsigned int route = i%hiveGlobalRankCount;
            guids[i] = hiveGuidCreateForRank(route, (unsigned int)type);
    //        if(route == hiveGlobalRankId)
    //            hiveRouteTableAddItem(NULL, guids[i], hiveGlobalRankId, false);
        }
    }
    return guids;
}

hiveGuidRange * hiveNewGuidRangeNode(hiveType_t type, unsigned int size, unsigned int route)
{
    hiveGuidRange * range = NULL;
    if(size && type > HIVE_NULL && type < HIVE_LAST_TYPE)
    {
        range = hiveCalloc(sizeof(hiveGuidRange));
        range->size = size;
        range->startGuid = hiveGuidCreateForRankInternal(route, (unsigned int)type, size);        
//        if(hiveIsGuidLocalExt(range->startGuid))
//        {
//            hiveGuid temp = (hiveGuid) range->startGuid;
//            for(unsigned int i=0; i<size; i++)
//            {
//                hiveRouteTableAddItem(NULL, temp.bits, route);
//                temp.fields.key++;
//            }
//        }
    }
    return range;
}

hiveGuid_t hiveGetGuid(hiveGuidRange * range, unsigned int index)
{
    if(!range || index >= range->size)
    {
        return NULL_GUID;
    }
    hiveGuid ret = (hiveGuid)range->startGuid;
    ret.fields.key+=index;
    return ret.bits;
}

hiveGuid_t hiveGuidRangeNext(hiveGuidRange * range)
{
    hiveGuid_t ret = NULL_GUID;
    if(range)
    {
        if(range->index < range->size)
            ret = hiveGetGuid(range, range->index);
    }
    return ret;
}

bool hiveGuidRangeHasNext(hiveGuidRange * range)
{
    if(range)
        return (range->size < range->index);
    return false; 
}

void hiveGuidRangeResetIter(hiveGuidRange * range)
{
    if(range)
        range->index = 0;
}

bool hiveIsInGuidRange(hiveGuidRange * range, hiveGuid_t guid)
{
    hiveGuid startGuid = (hiveGuid) range->startGuid;
    hiveGuid toCheck = (hiveGuid) guid;
    
    if(startGuid.fields.rank != toCheck.fields.rank)
        return false;
    
    if(startGuid.fields.type != toCheck.fields.type)
        return false;
    
    if(startGuid.fields.key <= toCheck.fields.key && toCheck.fields.key < startGuid.fields.key + range->index)
        return true;
    
    return false;
}
