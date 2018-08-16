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

u64 * parallelStartKeys;
u64 * keys;
extern u64 globalGuidOn;

u64 * hiveGuidGeneratorGetKey(unsigned int route, unsigned int type )
{
    if(globalGuidOn && hiveGlobalRankId)
        return &parallelStartKeys[route*HIVE_LAST_TYPE + type];
    return &keys[route*HIVE_LAST_TYPE + type];
}

hiveGuid_t hiveGuidCreateForRankInternal(unsigned int route, unsigned int type, unsigned int guidCount)
{
    hiveGuid guid;
    u64 * key = hiveGuidGeneratorGetKey(route, type);
    if(globalGuidOn)
    {
        guid.fields.local = 1;
        guid.fields.thread = 0;
    }
    else
    {
        if(route == hiveGlobalRankId)
        {
            guid.fields.local = 1;
            guid.fields.thread = hiveThreadInfo.threadId;
        }
        else
        {
            guid.fields.local = 0;
            guid.fields.thread = hiveGlobalRankId;
        }
    }
    guid.fields.type = type;
    guid.fields.rank = route;
    guid.fields.key = (*key)+1;
    (*key)+=guidCount;
    DPRINTF("Key: %lu %lu %lu %lu %lu %p sizeof(hiveGuid): %u\n", guid.fields.local, guid.fields.type, guid.fields.rank, guid.fields.thread, guid.fields.key, guid.bits, sizeof(hiveGuid));
    HIVEEDTCOUNTERINCREMENTBY(guidAllocCounter, guidCount);
    return (hiveGuid_t)guid.bits;
}

hiveGuid_t hiveGuidCreateForRank(unsigned int route, unsigned int type)
{
    return hiveGuidCreateForRankInternal(route, type, 1);
}

void setGuidGeneratorAfterParallelStart()
{   
    for(unsigned int i=0; i<HIVE_LAST_TYPE; i++)
        keys[hiveGlobalRankId*HIVE_LAST_TYPE + i] = parallelStartKeys[hiveGlobalRankId*HIVE_LAST_TYPE + i];
    hiveFree(parallelStartKeys);
}

void hiveGuidKeyGeneratorInit()
{
    parallelStartKeys = hiveCalloc(sizeof(u64) * HIVE_LAST_TYPE * hiveGlobalRankCount);
    keys = hiveCalloc(sizeof(u64) * HIVE_LAST_TYPE * hiveGlobalRankCount);
}

hiveType_t hiveGuidCast(hiveGuid_t guid, hiveType_t type)
{
    hiveGuid addressInfo = (hiveGuid) guid;
    addressInfo.fields.type = (unsigned int) type;
    return guid;
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
    
    if(startGuid.fields.thread != toCheck.fields.thread)
        return false;
    
    if(startGuid.fields.key <= toCheck.fields.key && toCheck.fields.key < startGuid.fields.key + range->index)
        return true;
    
    return false;
}
