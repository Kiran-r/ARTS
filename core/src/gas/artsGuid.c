//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include "artsGuid.h"
#include "artsGlobals.h"
#include "artsDebug.h"
#include "artsCounter.h"

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

__thread uint64_t globalGuidThreadId = 0;
__thread uint64_t * keys;
uint64_t numTables = 0;
uint64_t keysPerThread = 0;
uint64_t globalGuidOn = 0;
uint64_t minGlobalGuidThread = 0;
uint64_t maxGlobalGuidThread = 0;

void setGlobalGuidOn()
{
    globalGuidOn = ((uint64_t)1) << 40;
}

uint64_t * artsGuidGeneratorGetKey(unsigned int route, unsigned int type )
{
    return &keys[route*ARTS_LAST_TYPE + type];
}

artsGuid_t artsGuidCreateForRankInternal(unsigned int route, unsigned int type, unsigned int guidCount)
{
    artsGuid guid;
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
            artsDebugGenerateSegFault();
        }
    }
    else
    {
        
        uint64_t * key = artsGuidGeneratorGetKey(route, type);
        uint64_t value = *key;
        if(value + guidCount < keysPerThread)
        {
            guid.fields.key = value + keysPerThread * globalGuidThreadId;
            (*key)+=guidCount;
        }
        else
        {
            PRINTF("Out of guid keys\n");
            artsDebugGenerateSegFault();
        }
    }
    guid.fields.type = type;
    guid.fields.rank = route;
    DPRINTF("Key: %lu %lu %lu %p %lu sizeof(artsGuid): %u parallel start: %u\n", guid.fields.type, guid.fields.rank, guid.fields.key, guid.bits, (artsGuid_t) guid.bits, sizeof(artsGuid), (globalGuidOn!=0));
    ARTSEDTCOUNTERINCREMENTBY(guidAllocCounter, guidCount);
    return (artsGuid_t)guid.bits;
}

artsGuid_t artsGuidCreateForRank(unsigned int route, unsigned int type)
{
    return artsGuidCreateForRankInternal(route, type, 1);
}

void setGuidGeneratorAfterParallelStart()
{
        unsigned int numOfTables = artsNodeInfo.workerThreadCount + 1;
        keysPerThread = globalGuidOn / (numOfTables * artsGlobalRankCount);
        DPRINTF("Keys per thread %lu\n", keysPerThread);
        globalGuidOn = 0;
}

void artsGuidKeyGeneratorInit()
{
    numTables           = (artsGlobalRankCount == 1) ? artsNodeInfo.workerThreadCount : artsNodeInfo.workerThreadCount + 1;
    uint64_t localId         = (artsThreadInfo.worker) ? artsThreadInfo.threadId : artsNodeInfo.workerThreadCount;
    minGlobalGuidThread = numTables * artsGlobalRankId;
    maxGlobalGuidThread = (artsGlobalRankCount == 1) ? minGlobalGuidThread + numTables : minGlobalGuidThread + numTables -1;
    globalGuidThreadId  = minGlobalGuidThread + localId;
    
//    PRINTF("numTables: %lu localId: %lu minGlobalGuidThread: %lu maxGlobalGuidThread: %lu globalGuidThreadId: %lu\n", numTables, localId, minGlobalGuidThread, maxGlobalGuidThread, globalGuidThreadId);
    keys = artsMalloc(sizeof(uint64_t) * ARTS_LAST_TYPE * artsGlobalRankCount);
    for(unsigned int i=0; i<ARTS_LAST_TYPE * artsGlobalRankCount; i++)
        keys[i] = 1;
}

artsGuid_t artsGuidCast(artsGuid_t guid, artsType_t type)
{
    artsGuid addressInfo = (artsGuid) guid;
    addressInfo.fields.type = (unsigned int) type;
    return (artsGuid_t) addressInfo.bits;
}

artsType_t artsGuidGetType(artsGuid_t guid)
{
    artsGuid addressInfo = (artsGuid) guid;
    return addressInfo.fields.type;
}

unsigned int artsGuidGetRank(artsGuid_t guid)
{
    artsGuid addressInfo = (artsGuid) guid;
    return addressInfo.fields.rank;
}

bool artsIsGuidLocal(artsGuid_t guid)
{
    return (artsGlobalRankId == artsGuidGetRank(guid));
}

artsGuid_t artsReserveGuidRoute(artsType_t type, unsigned int route)
{
    artsGuid_t guid = NULL_GUID;
    route = route % artsGlobalRankCount;
    if(type > ARTS_NULL && type < ARTS_LAST_TYPE)
        guid = artsGuidCreateForRankInternal(route, (unsigned int)type, 1);
//    if(route == artsGlobalRankId)
//        artsRouteTableAddItem(NULL, guid, artsGlobalRankId, false);    
    return guid;
}

artsGuid_t * artsReserveGuidsRoundRobin(unsigned int size, artsType_t type)
{
    artsGuid_t * guids = NULL;
    if(type > ARTS_NULL && type < ARTS_LAST_TYPE)
    {
        artsGuid_t * guids = (artsGuid_t*) artsMalloc(size*sizeof(artsGuid_t));
        for(unsigned int i=0; i<size; i++)
        {
            unsigned int route = i%artsGlobalRankCount;
            guids[i] = artsGuidCreateForRank(route, (unsigned int)type);
    //        if(route == artsGlobalRankId)
    //            artsRouteTableAddItem(NULL, guids[i], artsGlobalRankId, false);
        }
    }
    return guids;
}

artsGuidRange * artsNewGuidRangeNode(artsType_t type, unsigned int size, unsigned int route)
{
    artsGuidRange * range = NULL;
    if(size && type > ARTS_NULL && type < ARTS_LAST_TYPE)
    {
        range = artsCalloc(sizeof(artsGuidRange));
        range->size = size;
        range->startGuid = artsGuidCreateForRankInternal(route, (unsigned int)type, size);        
//        if(artsIsGuidLocalExt(range->startGuid))
//        {
//            artsGuid temp = (artsGuid) range->startGuid;
//            for(unsigned int i=0; i<size; i++)
//            {
//                artsRouteTableAddItem(NULL, temp.bits, route);
//                temp.fields.key++;
//            }
//        }
    }
    return range;
}

artsGuid_t artsGetGuid(artsGuidRange * range, unsigned int index)
{
    if(!range || index >= range->size)
    {
        return NULL_GUID;
    }
    artsGuid ret = (artsGuid)range->startGuid;
    ret.fields.key+=index;
    return ret.bits;
}

artsGuid_t artsGuidRangeNext(artsGuidRange * range)
{
    artsGuid_t ret = NULL_GUID;
    if(range)
    {
        if(range->index < range->size)
            ret = artsGetGuid(range, range->index);
    }
    return ret;
}

bool artsGuidRangeHasNext(artsGuidRange * range)
{
    if(range)
        return (range->size < range->index);
    return false; 
}

void artsGuidRangeResetIter(artsGuidRange * range)
{
    if(range)
        range->index = 0;
}

bool artsIsInGuidRange(artsGuidRange * range, artsGuid_t guid)
{
    artsGuid startGuid = (artsGuid) range->startGuid;
    artsGuid toCheck = (artsGuid) guid;
    
    if(startGuid.fields.rank != toCheck.fields.rank)
        return false;
    
    if(startGuid.fields.type != toCheck.fields.type)
        return false;
    
    if(startGuid.fields.key <= toCheck.fields.key && toCheck.fields.key < startGuid.fields.key + range->index)
        return true;
    
    return false;
}
