#include <string.h>
#include "hiveArrayDb.h"
#include "hiveGlobals.h"
#include "hiveGuid.h"
#include "hiveMalloc.h"
#include "hiveDbFunctions.h"
#include "hiveEdtFunctions.h"
#include "hiveRemoteFunctions.h"

unsigned int hiveGetSizeArrayDb(hiveArrayDb_t * array)
{
    return array->elementsPerBlock * array->numBlocks;
}

void * copyDb(void * ptr, unsigned int size, hiveGuid_t guid)
{
    struct hiveDb * db = ((struct hiveDb *)ptr) - 1;
    struct hiveDb * newDb = hiveCalloc(size);
    memcpy(newDb, db, size);
    newDb->guid = guid;
    return (void*) (newDb+1);
}

hiveArrayDb_t * hiveNewArrayDbWithGuid(hiveGuid_t guid, unsigned int elementSize, unsigned int numElements)
{
    unsigned int numBlocks = hiveGlobalRankCount;
    unsigned int elementsPerBlock = numElements / numBlocks;
    if(!elementsPerBlock)
    {
        elementsPerBlock = 1;
        numBlocks = numElements;
    }
    
    unsigned int allocSize = sizeof(hiveArrayDb_t) + elementSize * elementsPerBlock;
    hiveArrayDb_t * block = NULL;
    if(numBlocks)
    {
        block = hiveDbCreateWithGuid(guid, allocSize, true);
        block->elementSize = elementSize;
        block->elementsPerBlock = elementsPerBlock;
        block->numBlocks = numBlocks;
        
        struct hiveDb * toSend = ((struct hiveDb *)block) - 1;
        for(unsigned int i=0; i<hiveGlobalRankCount; i++)
        {
            if(i!=hiveGlobalRankId)
            {
                hiveRemoteMemoryMoveNoFree(i, guid, toSend, allocSize + sizeof(struct hiveDb), HIVE_REMOTE_DB_MOVE_MSG);
            }
        }
    }
    return block;
}

hiveGuid_t hiveNewArrayDb(hiveArrayDb_t **addr, unsigned int elementSize, unsigned int numElements)
{
    hiveGuid_t guid = hiveReserveGuidRoute(HIVE_DB, hiveGlobalRankId);
    *addr = hiveNewArrayDbWithGuid(guid, elementSize, numElements);
    return guid;
}

hiveGuid_t getArrayDbGuid(hiveArrayDb_t * array)
{
    struct hiveDb * db = ((struct hiveDb *)array) - 1;
    return db->guid;
}

unsigned int getOffsetFromIndex(hiveArrayDb_t * array, unsigned int index)
{    
    unsigned int base = sizeof(hiveArrayDb_t);
    unsigned int local = (index % array->elementsPerBlock) * array->elementSize;
    return base + local;
}

unsigned int getRankFromIndex(hiveArrayDb_t * array, unsigned int index)
{
    return index/array->elementsPerBlock;
}

void hiveGetFromArrayDb(hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index)
{
    hiveGuid_t guid = getArrayDbGuid(array);
    unsigned int rank = getRankFromIndex(array, index);
    unsigned int offset = getOffsetFromIndex(array, index);
    PRINTF("Getting i: %u From Rank: %u\n", index, rank);
    hiveGetFromDbAt(edtGuid, guid, slot, offset, array->elementSize, rank);
}

void hivePutInArrayDb(void * ptr, hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index)
{
    hiveGuid_t guid = getArrayDbGuid(array);
    unsigned int rank = getRankFromIndex(array, index);
    unsigned int offset = getOffsetFromIndex(array, index);
    hivePutInDbAt(ptr, edtGuid, guid, slot, offset, array->elementSize, rank);
}

void hiveForEachInArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, u32 paramc, u64 * paramv)
{
    u64 * args = hiveMalloc(sizeof(u64) * (paramc+1));
    memcpy(&args[1], paramv, sizeof(u64) * paramc);
    
    unsigned int size = hiveGetSizeArrayDb(array);
    for(unsigned int i=0; i<size; i++)
    {
        args[0] = i;
        unsigned int route = getRankFromIndex(array, i);
        hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc+1, args, 1);
        hiveGetFromArrayDb(guid, 0, array, i);
    }
}

hiveGuid_t hiveGatherArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u64 depc)
{
    unsigned int offset = getOffsetFromIndex(array, 0);
    unsigned int size = array->elementSize * array->elementsPerBlock;
    hiveGuid_t arrayGuid = getArrayDbGuid(array);
    
    hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc, paramv, array->numBlocks + depc);
    for(unsigned int i=0; i<array->numBlocks; i++)
    {
        hiveGetFromDbAt(guid, arrayGuid, i, offset, size, i);
    }
}

hiveGuid_t loopPolicy(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveEdt_t funcPtr = (hiveEdt_t)paramv[0];
    unsigned int stride = paramv[1];
    unsigned int end = paramv[2];
    unsigned int start = paramv[3];
    
    hiveArrayDb_t * array = depv[0].ptr;
    unsigned int offset = getOffsetFromIndex(array, start);
    char * raw = depv[0].ptr;
    
    for(unsigned int i=start; i<end; i+=stride)
    {
        paramv[3] = i;
        depv[0].ptr = (void*)(&raw[offset]);
        funcPtr(paramc-3, &paramv[3], 1, depv);
        offset+=array->elementSize;
    }
    depv[0].ptr = (void*)raw;
}

void hiveForEachInArrayDbAtData(hiveArrayDb_t * array, unsigned int stride, hiveEdt_t funcPtr, u32 paramc, u64 * paramv)
{
    unsigned int blockSize = array->elementsPerBlock;
    unsigned int size = hiveGetSizeArrayDb(array);
    if(size%stride)
    {
        PRINTF("WARNING: Size is not divisible by stride!");
    }
    hiveGuid_t guid = getArrayDbGuid(array);
    u64 * args = hiveMalloc(sizeof(u64) * (paramc+4));
    memcpy(&args[4], paramv, sizeof(u64) * paramc);
    args[0] = (u64)funcPtr;
    args[1] = stride;
    for(unsigned int i=0; i<size; i+=blockSize)
    {
        args[2] = (i+blockSize < size) ? i+blockSize : size;
        args[3] = i;
        hiveActiveMessageWithDbAt(loopPolicy, paramc+4, args, 0, guid, getRankFromIndex(array, i));
    }
}