#include <string.h>
#include "hiveArrayDb.h"
#include "hiveGlobals.h"
#include "hiveGuid.h"
#include "hiveMalloc.h"
#include "hiveDbFunctions.h"
#include "hiveEdtFunctions.h"
#include "hiveRemoteFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveTerminationDetection.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveAtomics.h"

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

hiveGuid_t hiveNewArrayDb(hiveArrayDb_t **addr, unsigned int elementSize, unsigned int elementsPerBlock, unsigned int numBlocks)
{
    unsigned int allocSize = sizeof(hiveArrayDb_t) + numBlocks * sizeof(hiveGuid_t) + elementSize * elementsPerBlock;
    hiveArrayDb_t * block = NULL;
    hiveGuid_t guid = NULL_GUID;
    if(numBlocks)
    {
        guid = hiveDbCreate((void**)&block, allocSize, true);

        block->blockId = 0;
        block->elementSize = elementSize;
        block->elementsPerBlock = elementsPerBlock;
        block->numBlocks = numBlocks;
        hiveGuid_t * list = (hiveGuid_t *) &block->head[0];
        list[0] = guid;
        
        unsigned int current = hiveGlobalRankId+1;
        for(unsigned int i=1; i<numBlocks; i++, current++)
            list[i] = hiveGuidCreateForRank(current % hiveGlobalRankCount, HIVE_DB);
        
        current = hiveGlobalRankId+1;
        for(unsigned int i=1; i<numBlocks; i++, current++)
        {
            if(!hiveDbCreateWithGuidAndData(list[i], block, allocSize, true))
            {
                hiveArrayDb_t * ptr = copyDb((void*)block, allocSize + sizeof(struct hiveDb), list[i]);
                ptr->blockId = i;
                struct hiveDb * toSend = ((struct hiveDb *)ptr) - 1;
                hiveRemoteMemoryMove(current % hiveGlobalRankCount, list[i], toSend, allocSize + sizeof(struct hiveDb), HIVE_REMOTE_DB_MOVE_MSG, hiveFree);
            }
        }
    }
    *addr = block;
    return guid;
}

unsigned int getOffsetFromIndex(hiveArrayDb_t * array, unsigned int index)
{    
    unsigned int base = sizeof(hiveArrayDb_t) + sizeof(hiveGuid_t)*array->numBlocks;
    unsigned int local = (index % array->elementsPerBlock) * array->elementSize;
    return base + local;
}

hiveGuid_t getGuidFromIndex(hiveArrayDb_t * array, unsigned int index)
{
    hiveGuid_t * list = (hiveGuid_t *)array->head;
    return list[index/array->elementsPerBlock];
}

void hiveGetFromArrayDb(hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index)
{
    hiveGuid_t guid = getGuidFromIndex(array, index);
    unsigned int offset = getOffsetFromIndex(array, index);
    hiveGetFromDb(edtGuid, guid, slot, offset, array->elementSize);
}

void hivePutInArrayDb(void * ptr, hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index)
{
    hiveGuid_t guid = getGuidFromIndex(array, index);
    unsigned int offset = getOffsetFromIndex(array, index);
    hivePutInDb(ptr, edtGuid, guid, slot, offset, array->elementSize);
}

void hiveForEachInArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, u32 paramc, u64 * paramv)
{
    u64 * args = hiveMalloc(sizeof(u64) * (paramc+1));
    memcpy(&args[1], paramv, sizeof(u64) * paramc);
    
    unsigned int size = hiveGetSizeArrayDb(array);
    for(unsigned int i=0; i<size; i++)
    {
        args[0] = i;
        unsigned int route = hiveGuidGetRank(getGuidFromIndex(array, i));
        hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc+1, args, 1);
        hiveGetFromArrayDb(guid, 0, array, i);
    }
}

hiveGuid_t hiveGatherArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u64 depc)
{
    unsigned int offset = getOffsetFromIndex(array, 0);
    unsigned int size = array->elementSize * array->elementsPerBlock;
    hiveGuid_t * list = (hiveGuid_t *) &array->head[0];
    
    hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc, paramv, array->numBlocks + depc);
    for(unsigned int i=0; i<array->numBlocks; i++)
    {
        hiveGetFromDb(guid, list[i], i, offset, size);
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

void hiveForEachInArrayDbAtData(hiveArrayDb_t * array, unsigned int stride, unsigned int blockSize, hiveEdt_t funcPtr, u32 paramc, u64 * paramv)
{
    unsigned int size = hiveGetSizeArrayDb(array);
    if(size%stride)
    {
        PRINTF("WARNING: Size is not divisible by stride!");
    }
    
    u64 * args = hiveMalloc(sizeof(u64) * (paramc+4));
    memcpy(&args[4], paramv, sizeof(u64) * paramc);
    args[0] = (u64)funcPtr;
    args[1] = stride;
    for(unsigned int i=0; i<size; i+=blockSize)
    {
        args[2] = (i+blockSize < size) ? i+blockSize : size;
        args[3] = i;
        hiveActiveMessageWithDb(loopPolicy, paramc+4, args, 0, getGuidFromIndex(array, i));
    }
}

void internalAtomicAddInArrayDb(hiveGuid_t dbGuid, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
    if(db)
    {
        hiveArrayDb_t * array = (hiveArrayDb_t*)(db+1);
        //Do this so when we increment finished we can check the term status
        incrementQueueEpoch(epochGuid);
        
        unsigned int offset = getOffsetFromIndex(array, index);
        unsigned int * data = (unsigned int*)(((char*) array) + offset);
        unsigned int result = hiveAtomicAdd(data, toAdd);
//        PRINTF("index: %u result: %u\n", index, result);
        
        if(edtGuid)
        {
//            PRINTF("Signaling edtGuid: %lu\n", edtGuid);
            hiveSignalEdt(edtGuid, result, slot, DB_MODE_SINGLE_VALUE);
        }

        incrementFinishedEpoch(epochGuid);
    }
    else
    {
        hiveOutOfOrderAtomicAddInArrayDb(dbGuid, index, toAdd, edtGuid, slot, epochGuid);
    }
}

void hiveAtomicAddInArrayDb(hiveArrayDb_t * array, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot)
{
    hiveGuid_t dbGuid = getGuidFromIndex(array, index);
    hiveGuid_t epochGuid = hiveGetCurrentEpochGuid();
    incrementActiveEpoch(epochGuid);
    unsigned int rank = hiveGuidGetRank(dbGuid);
    if(rank==hiveGlobalRankId)
        internalAtomicAddInArrayDb(dbGuid, index, toAdd, edtGuid, slot, epochGuid);
    else
        hiveRemoteAtomicAddInArrayDb(rank, dbGuid, index, toAdd, edtGuid, slot, epochGuid);
}

void internalAtomicCompareAndSwapInArrayDb(hiveGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
    if(db)
    {
        hiveArrayDb_t * array = (hiveArrayDb_t*)(db+1);
        //Do this so when we increment finished we can check the term status
        incrementQueueEpoch(epochGuid);
        
        unsigned int offset = getOffsetFromIndex(array, index);
        unsigned int * data = (unsigned int*)(((char*) array) + offset);
        unsigned int result = hiveAtomicCswap(data, oldValue, newValue);
//        PRINTF("index: %u result: %u\n", index, result);
        
        if(edtGuid)
        {
//            PRINTF("Signaling edtGuid: %lu\n", edtGuid);
            hiveSignalEdt(edtGuid, result, slot, DB_MODE_SINGLE_VALUE);
        }

        incrementFinishedEpoch(epochGuid);
    }
    else
    {
        hiveOutOfOrderAtomicCompareAndSwapInArrayDb(dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
    }
}

void hiveAtomicCompareAndSwapInArrayDb(hiveArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot)
{
    hiveGuid_t dbGuid = getGuidFromIndex(array, index);
    hiveGuid_t epochGuid = hiveGetCurrentEpochGuid();
    incrementActiveEpoch(epochGuid);
    unsigned int rank = hiveGuidGetRank(dbGuid);
    if(rank==hiveGlobalRankId)
        internalAtomicCompareAndSwapInArrayDb(dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
    else
        hiveRemoteAtomicCompareAndSwapInArrayDb(rank, dbGuid, index, oldValue, newValue, edtGuid, slot, epochGuid);
}