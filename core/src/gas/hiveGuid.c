#include "hive.h"
#include "hiveGuid.h"
#include "hiveHash.h"
#include "hiveRouteTable.h"
#include "hiveGlobals.h"
#include "hiveMalloc.h"
#include "hiveCounter.h"
#include <stdint.h>
#include <string.h>

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define ABSOLUTEMINGUID 1

__thread unsigned int guidKey=0;
__thread unsigned int routeKey[4]={0};
__thread unsigned int routeFastKey=0;

#ifndef NOPARALLEL
extern u64 globalGuidOn;
#endif

struct keyGenerator
{
    unsigned int size;
    void * next;
    u64 key[];
};

__thread struct keyGenerator * getKey = NULL;

u64 hiveGuidMin;
u64 hiveGuidMax;
struct hiveRouteTable *  hiveGlobalRouteTable;

u64 * hiveGuidGeneratorGetKey( unsigned int route, unsigned int type )
{
    /* TODO: Only allocate space for the node you are running on
     * Currently entire key space is allocate, but this might not be scalable. 
     * An alternative is to allocate per route when upon first seeing the route.
     */
    return &(getKey->key[route*4+type]);
}

hiveGuid_t hiveGuidCreateForRankInternal( unsigned int route, unsigned int type, unsigned int guidCount)
{
    hiveGuid guid;
    u64 * key;
#ifndef NOPARALLEL
    if(globalGuidOn)
    {
        key = &globalGuidOn;
    }
    else
#endif
    key = hiveGuidGeneratorGetKey( route, type );
    guid.fields.isLocal = 0;
    guid.fields.rank = route;
    //only thread 0 should operate with globalGuidOn = 1 
    guid.fields.thread = hiveThreadInfo.threadId;
    u64 temp = (*key)+guidCount;
    if(temp >= 17179869183)
        PRINTF("Hit 2^34-1 guids!!!\n", temp);
    guid.fields.key = (*key)+1;
    (*key)+=guidCount;
    guid.fields.type = type;
//    PRINTF("Key: %lu %lu %lu %lu %lu %p\n", guid.fields.isLocal, guid.fields.rank, guid.fields.thread, guid.fields.key, guid.fields.type, guid.bits);
    HIVEEDTCOUNTERINCREMENTBY(guidAllocCounter, guidCount);
    return (hiveGuid_t)guid.bits;
}

hiveGuid_t hiveGuidCreateForRank( unsigned int route, unsigned int type )
{
    return hiveGuidCreateForRankInternal(route, type, 1);
}

void setGuidGeneratorAfterParallelStart()
{
    u64 * key = hiveGuidGeneratorGetKey( 0, 0 );
    (*key) = globalGuidOn;
    key = hiveGuidGeneratorGetKey( 0, 1 );
    (*key) = globalGuidOn;
    key = hiveGuidGeneratorGetKey( 0, 2 );
    (*key) = globalGuidOn;
    key = hiveGuidGeneratorGetKey( 0, 3 );
    (*key) = globalGuidOn;
}

hiveGuid_t hiveGuidCreateFastPathLocal( unsigned int route, void * address, unsigned int type  )
{
    //void * fastLocalPath = hiveRouteTableCreateLocalEntry( address, hiveGlobalRankId );
    
    hiveGuid_t fakeGuid = hiveGuidCreateForRank( hiveGlobalRankId, ((struct hiveHeader *)address)->type);    
    void * fastLocalPath = hiveRouteTableAddItem( address, (hiveGuid_t) fakeGuid, hiveGlobalRankId, false );
    hiveGuid guid;
    guid.local.isLocal = 1;
    guid.local.rank = route;
    guid.local.type = type;
    guid.local.addr = ((uintptr_t)fastLocalPath) >> 7;
    
    //PRINTF("%p %lx %lx\n", fastLocalPath, ((uintptr_t)fastLocalPath) >> 7,  guid.local.addr);

    return (hiveGuid_t) guid.bits;
}

void * hiveGuidGetFastPathLocal( hiveGuid_t guid )
{
    
    hiveGuid bitInfo = ((hiveGuid)guid);
    if( bitInfo.local.rank == hiveGlobalRankId && bitInfo.local.isLocal )
    {
        uintptr_t address = (uintptr_t) bitInfo.local.addr;
        
        address <<= 7;
      
        if( (address & 0x800000000000) != 0 )
        {
            PRINTF("EXTEND\n");
            address |= 0xFFFF000000000000;
        }
        //PRINTF("%lx r %p\n", (uintptr_t)bitInfo.local.addr, (void *)address);
        return (void *) address;

    }
    else
    {
        return NULL;
    }
}

hiveGuid_t hiveGuidCreate(void * address)
{
    hiveGuid_t guid;
    if(globalGuidOn)
    {
        guid = hiveGuidCreateForRank(0, ((struct hiveHeader *)address)->type);
//        hiveRouteTableAddItem(address, (hiveGuid_t) guid, hiveGlobalRankId, false);
    }
    else
    {
        //guid = hiveGuidCreateFastPathLocal( hiveGlobalRankId, address, ((struct hiveHeader *)address)->type);
        guid = hiveGuidCreateForRank(hiveGlobalRankId, ((struct hiveHeader *)address)->type);    
        hiveRouteTableAddItem(address, (hiveGuid_t) guid, hiveGlobalRankId, false);
    }
    return guid;
}

void hiveGuidKeyGeneratorInit()
{
    //multiplied by 4 for each type (i.e. edt, event, DB)
    getKey = hiveMalloc(sizeof(struct keyGenerator) + sizeof(u64) * 4 * hiveGlobalRankCount);
    getKey->size = hiveGlobalRankCount;
    getKey->next  = NULL;
    for(int i=0; i< hiveGlobalRankCount * 4; i++)
    {
        getKey->key[i] = hiveGuidMin;
    }
}

void hiveGuidTableInit( unsigned int routeInitSize  )
{
    
    u64 guidsPerRank = GUID_MAX_KEYS/hiveGlobalRankCount;
    u64 guidsPerRankRem = GUID_MAX_KEYS%hiveGlobalRankCount;
    
    if( hiveGlobalRankId >= guidsPerRankRem )
    {
        hiveGuidMin = (guidsPerRank+1)* guidsPerRankRem + (guidsPerRank * ( hiveGlobalRankId - guidsPerRankRem  )  ) ;
        
        if(hiveGuidMin == 0)
            hiveGuidMin = ABSOLUTEMINGUID;

        hiveGuidMax = hiveGuidMin + guidsPerRank;
    } 
    else
    {
        hiveGuidMin = (guidsPerRank+1)* hiveGlobalRankId;
        
        hiveGuidMax = hiveGuidMin + guidsPerRank + 1;
    }
    DPRINTF( "sdsdsds %d %ld %ld %ld %d %ld\n", hiveNodeInfo.totalThreadCount, hiveGuidMin, hiveGuidMax, GUID_MAX_KEYS, hiveGlobalRankCount, guidsPerRank); 

    int size = 1;
    int i;
}

u32 hiveGuidGetType( hiveGuid_t guid )
{
    hiveGuid addressInfo = (hiveGuid) guid;
    return addressInfo.fields.type;
}

unsigned int hiveGuidGetRank( hiveGuid_t guid )
{
    hiveGuid addressInfo = (hiveGuid) guid;
    return addressInfo.fields.rank;
}

bool hiveIsGuidLocal(hiveGuid_t guid)
{
    return (hiveGlobalRankId == hiveGuidGetRank(guid));
}

hiveGuid_t hiveReserveGuidRoute(unsigned int type, unsigned int route)
{
    route = route % hiveGlobalRankCount;
    hiveGuid_t guid = hiveGuidCreateForRank(route, type);
//    if(route == hiveGlobalRankId)
//        hiveRouteTableAddItem(NULL, guid, hiveGlobalRankId, false);    
    return guid;
}

hiveGuid_t * hiveReserveGuidsRoundRobin(unsigned int size, unsigned int type)
{
    hiveGuid_t * guids = (hiveGuid_t*) hiveMalloc(size*sizeof(hiveGuid_t));
    for(unsigned int i=0; i<size; i++)
    {
        unsigned int route = i%hiveGlobalRankCount;
        guids[i] = hiveGuidCreateForRank(route, type);
//        if(route == hiveGlobalRankId)
//            hiveRouteTableAddItem(NULL, guids[i], hiveGlobalRankId, false);
    }
    return guids;
}

hiveGuidRange * hiveNewGuidRangeNode(unsigned int type, unsigned int size, unsigned int route)
{
    hiveGuidRange * range = NULL;
    if(size)
    {
        range = hiveCalloc(sizeof(hiveGuidRange));
        range->size = size;
        range->startGuid = hiveGuidCreateForRankInternal(route, type, size);        
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
    
    if(startGuid.fields.isLocal != toCheck.fields.isLocal)
        return false;
    
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

