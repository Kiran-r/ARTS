#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"
#include "hiveAtomics.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveGlobals.h"
#include "hiveRemoteFunctions.h"
#include "hiveRouteTable.h"
#include "hiveArrayList.h"
#include "hiveDebug.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

#define EpochMask   0x7FFFFFFFFFFFFFFF  
#define EpochBit 0x8000000000000000

#define DEFAULT_EPOCH_POOL_SIZE 4096
__thread hiveEpochPool_t * epochThreadPool;

void globalShutdownGuidIncActive()
{
    if(hiveNodeInfo.shutdownEpoch)
        incrementActiveEpoch(hiveNodeInfo.shutdownEpoch);
}

void globalShutdownGuidIncQueue()
{
    if(hiveNodeInfo.shutdownEpoch)
        incrementQueueEpoch(hiveNodeInfo.shutdownEpoch);
}

void globalShutdownGuidIncFinished()
{
    if(hiveNodeInfo.shutdownEpoch)
        incrementFinishedEpoch(hiveNodeInfo.shutdownEpoch);
}

void globalGuidShutdown(hiveGuid_t guid)
{
    if(hiveNodeInfo.shutdownEpoch == guid)
    {
        DPRINTF("TERMINATION GUID SHUTDOWN %lu\n", guid);
        hiveShutdown();
    }
}

bool decrementQueueEpoch(hiveEpoch_t * epoch)
{
    DPRINTF("Dec queue Epoch: %lu\n", epoch->guid);
    u64 local;
    while(1)
    {
        local = epoch->queued;
        if(local == 1)
        {
            if(1 == hiveAtomicCswapU64(&epoch->queued, 1, EpochBit))
                return true;
        }
        else
        {
            if(local == hiveAtomicCswapU64(&epoch->queued, local, local - 1))
                return false;
        }
    }
}

void incrementQueueEpoch(hiveGuid_t epochGuid)
{
    DPRINTF("Inc queue Epoch: %lu\n", epochGuid);
    if(epochGuid != NULL_GUID)
    {
        hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
        if(epoch)
        {
            hiveAtomicAddU64(&epoch->queued, 1);
        }
        else
        {
            hiveOutOfOrderIncQueueEpoch(epochGuid);
            DPRINTF("OOIncQueueEpoch %lu\n", epochGuid);
        }
    }
}

void incrementActiveEpoch(hiveGuid_t epochGuid)
{
    DPRINTF("Inc active Epoch: %lu\n", epochGuid);
    hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
    if(epoch)
    {
        hiveAtomicAdd(&epoch->activeCount, 1);
    }
    else
    {
        hiveOutOfOrderIncActiveEpoch(epochGuid);
        DPRINTF("OOIncActiveEpoch %lu\n", epochGuid);
    }
}

void incrementFinishedEpoch(hiveGuid_t epochGuid)
{
    DPRINTF("In finish Epoch: %lu\n", epochGuid);
    if(epochGuid != NULL_GUID)
    {
        hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
        if(epoch)
        {
            hiveAtomicAdd(&epoch->finishedCount, 1);
            if(hiveGlobalRankCount == 1)
            {
                if(!checkEpoch(epoch, epoch->activeCount, epoch->finishedCount))
                {
                    if(epoch->phase == PHASE_3)
                        deleteEpoch(epochGuid, epoch);
                }
            }
            else
            {
                unsigned int rank = hiveGuidGetRank(epochGuid);
                if(rank == hiveGlobalRankId)
                {
                    if(!hiveAtomicSubU64(&epoch->queued, 1))
                    {
                        if(!hiveAtomicCswapU64(&epoch->outstanding, 0, hiveGlobalRankCount))
                        {
                            broadcastEpochRequest(epochGuid);
                            DPRINTF("%lu Broadcasting req... \n", epochGuid);
                        }
                        else
                        {
                            DPRINTF("OUTSTANDING: %lu %lu\n", epoch->guid, epoch->outstanding);
                        }
                    }
                    else
                    {
                        DPRINTF("QUEUED: %lu %lu\n", epoch->guid, epoch->outstanding);
                    }
                }
                else
                {
                    if(decrementQueueEpoch(epoch))
                    {
                        hiveRemoteEpochSend(rank, epochGuid, epoch->activeCount, epoch->finishedCount);
                        DPRINTF("%lu Now responding... \n", epochGuid);
                    }
                }  
            }
        }
        else
        {
            hiveOutOfOrderIncFinishedEpoch(epochGuid);
            DPRINTF("%lu ooFinish\n", epochGuid);
        }
    }
}

void sendEpoch(hiveGuid_t epochGuid, unsigned int source, unsigned int dest)
{
    hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
    if(epoch)
    {
        hiveAtomicFetchAndU64(&epoch->queued, EpochMask);
        if(!hiveAtomicCswapU64(&epoch->queued, 0, EpochBit))
        {
            hiveRemoteEpochSend(dest, epochGuid, epoch->activeCount, epoch->finishedCount);
//            PRINTF("%lu Sending Now...\n", epochGuid);
        }
//        else
//            PRINTF("Buffer Send...\n");
    }
    else
        hiveOutOfOrderSendEpoch(epochGuid, source, dest);
}

hiveEpoch_t * createEpoch(hiveGuid_t * guid, hiveGuid_t edtGuid, unsigned int slot)
{
    if(*guid == NULL_GUID)
        *guid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_EDT);
    
    hiveEpoch_t * epoch = hiveCalloc(sizeof(hiveEpoch_t));
    epoch->phase = PHASE_1;
    epoch->terminationExitGuid = edtGuid;
    epoch->terminationExitSlot = slot;
    epoch->guid = *guid;
    epoch->poolGuid = NULL_GUID;
    epoch->queued = (hiveIsGuidLocal(*guid)) ? 0 : EpochBit;
    hiveRouteTableAddItemRace(epoch, *guid, hiveGlobalRankId, false);
    hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler);
//    PRINTF("Create %lu %p\n", *guid, epoch);
    return epoch;
}

bool createShutdownEpoch()
{
    if(hiveNodeInfo.shutdownEpoch)
    {
        hiveNodeInfo.shutdownEpoch = hiveGuidCreateForRank(0, HIVE_EDT);
        hiveEpoch_t * epoch = createEpoch(&hiveNodeInfo.shutdownEpoch, NULL_GUID, 0);
        hiveAtomicAdd(&epoch->activeCount, hiveGetTotalWorkers());
        hiveAtomicAddU64(&epoch->queued, hiveGetTotalWorkers());
        DPRINTF("Shutdown guy %u : %lu --------> %lu %p\n", epoch->activeCount, epoch->queued, epoch->guid, epoch);
        return true;
    }
    return false;
}

void hiveAddEdtToEpoch(hiveGuid_t edtGuid, hiveGuid_t epochGuid)
{
    struct hiveEdt * edt = hiveRouteTableLookupItem(edtGuid);
    if(edt)
    {
        edt->epochGuid = epochGuid;
        incrementActiveEpoch(epochGuid);
        return;
    }
    DPRINTF("Out-of-order add to epoch not supported...\n");
    return;
}

void broadcastEpochRequest(hiveGuid_t epochGuid)
{
    unsigned int originRank = hiveGuidGetRank(epochGuid);
    for(unsigned int i=0; i<hiveGlobalRankCount; i++)
    {
        if(i != originRank)
        {
            hiveRemoteEpochReq(i, epochGuid);
        }
    }
}

hiveGuid_t hiveInitializeAndStartEpoch(hiveGuid_t finishEdtGuid, unsigned int slot)
{
//    hiveGuid_t guid = NULL_GUID;
//    hiveEpoch_t * epoch = createEpoch(&guid, finishEdtGuid, slot);
    hiveEpoch_t * epoch = getPoolEpoch(finishEdtGuid, slot);
    
    hiveSetCurrentEpochGuid(epoch->guid);
    hiveAtomicAdd(&epoch->activeCount, 1);
    hiveAtomicAddU64(&epoch->queued, 1);

//    for(unsigned int i=0; i<hiveGlobalRankCount; i++)
//    {
//        if(i != hiveGlobalRankId)
//            hiveRemoteEpochInitSend(i, guid, finishEdtGuid, slot);
//    }

    DPRINTF("%u : %lu --------> %lu %p\n", epoch->activeCount, epoch->queued, epoch->guid, epoch);
    return epoch->guid;
}

hiveGuid_t hiveInitializeEpoch(unsigned int rank, hiveGuid_t finishEdtGuid, unsigned int slot)
{
    hiveGuid_t guid = hiveGuidCreateForRank(rank, HIVE_EDT);
    createEpoch(&guid, finishEdtGuid, slot);
    if(!hiveNodeInfo.readyToExecute) {
        for(unsigned int i=0; i<hiveGlobalRankCount; i++)
        {
            if(i != hiveGlobalRankId)
                hiveRemoteEpochInitSend(i, guid, finishEdtGuid, slot);
        }
    }
    return guid;
}

void hiveStartEpoch(hiveGuid_t epochGuid) 
{
    hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
    if(epoch)
    {
        hiveSetCurrentEpochGuid(epoch->guid);
        hiveAtomicAdd(&epoch->activeCount, 1);
        hiveAtomicAddU64(&epoch->queued, 1);
    }
    else
        PRINTF("Out-of-Order epoch start not supported %lu\n", epochGuid);
}

bool checkEpoch(hiveEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish)
{
    unsigned int diff = totalActive - totalFinish;
    DPRINTF("%lu : %u - %u = %u\n", epoch->guid, totalActive, totalFinish, diff);
    //We have a zero
    if(totalFinish && !diff)
    {
        //Lets check the phase and if we have the same counts as before
        if(epoch->phase == PHASE_2 && epoch->lastActiveCount == totalActive && epoch->lastFinishedCount == totalFinish) 
        {
            epoch->phase = PHASE_3;
            DPRINTF("%lu epoch done!!!!!!!\n", epoch->guid);
            if(epoch->waitPtr)
                *epoch->waitPtr = 0;
            if(epoch->terminationExitGuid)
            {
                DPRINTF("%lu Calling finalization continuation provided by the user %u\n", epoch->guid, totalFinish);
                hiveSignalEdt(epoch->terminationExitGuid, totalFinish, epoch->terminationExitSlot, DB_MODE_SINGLE_VALUE);
            }
            else
            {
                globalGuidShutdown(epoch->guid);
            }
            return false;
        }
        else //We didn't match the last one so lets try again
        {
            epoch->lastActiveCount = totalActive;
            epoch->lastFinishedCount = totalFinish;
            epoch->phase = PHASE_2;
            DPRINTF("%lu Starting phase 2 %u\n", epoch->guid, epoch->lastFinishedCount);
            if(hiveGlobalRankCount == 1)
            {
                epoch->phase = PHASE_3;
                DPRINTF("%lu epoch done!!!!!!!\n", epoch->guid);
                if(epoch->waitPtr)
                    *epoch->waitPtr = 0;
                if(epoch->terminationExitGuid)
                {
                    DPRINTF("%lu Calling finalization continuation provided by the user %u !\n", epoch->guid, totalFinish);
                    hiveSignalEdt(epoch->terminationExitGuid, totalFinish, epoch->terminationExitSlot, DB_MODE_SINGLE_VALUE);
                }
                else
                {
                    globalGuidShutdown(epoch->guid);
                }
                return false;
            }
            else
                return true;
        }
    }
    else
        epoch->phase = PHASE_1;
    return (epoch->queued == 0);
}

void reduceEpoch(hiveGuid_t epochGuid, unsigned int active, unsigned int finish)
{
    hiveEpoch_t * epoch = hiveRouteTableLookupItem(epochGuid);
    if(epoch)
    {
        DPRINTF("%lu A: %u F: %u\n", epochGuid, active, finish);
        unsigned int totalActive = hiveAtomicAdd(&epoch->globalActiveCount, active);
        unsigned int totalFinish = hiveAtomicAdd(&epoch->globalFinishedCount, finish);
        if(hiveAtomicSubU64(&epoch->outstanding, 1) == 1)
        {
            DPRINTF("%lu A: %u F: %u\n", epochGuid, epoch->activeCount, epoch->finishedCount);
            totalActive+=epoch->activeCount;
            totalFinish+=epoch->finishedCount;
            
            //Reset for the next round
            epoch->globalActiveCount = 0;
            epoch->globalFinishedCount = 0;
            
            if(checkEpoch(epoch, totalActive, totalFinish))
            {
                DPRINTF("%lu REDUCE SEND\n", epochGuid);
                hiveAtomicAddU64(&epoch->outstanding, hiveGlobalRankCount-1);
                broadcastEpochRequest(epochGuid);
                //A better idea will be to know when to kick off a new round
                //the checkinCount == 0 indicates there is a new round can be kicked off
//                hiveAtomicSub(&epoch->checkinCount, 1);
            }
            else
                hiveAtomicSubU64(&epoch->outstanding, 1);
            
            if(epoch->phase == PHASE_3)
                deleteEpoch(epochGuid, epoch);
            
            DPRINTF("%lu EPOCH QUEUEU: %u\n", epochGuid, epoch->queued);
        }
        DPRINTF("###### %lu -> %lu\n", epoch->guid, epoch->outstanding);
    }
    else
        PRINTF("%lu ERROR: NO EPOCH\n", epochGuid);
}

hiveEpochPool_t * createEpochPool(hiveGuid_t * epochPoolGuid, unsigned int poolSize, hiveGuid_t * startGuid)
{
    if(*epochPoolGuid == NULL_GUID)
        *epochPoolGuid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_EDT);
    
    bool newRange = (*startGuid == NULL_GUID);
    hiveGuidRange temp;
    hiveGuidRange * range;
    if(newRange)
    {
        range = hiveNewGuidRangeNode(HIVE_EDT, poolSize, hiveGlobalRankId);
        *startGuid = hiveGetGuid(range, 0);
    }
    else
    {
        temp.size = poolSize;
        temp.index = 0;
        temp.startGuid = *startGuid;
        range = &temp;
    }
    
    
    
    hiveEpochPool_t * epochPool = hiveCalloc(sizeof(hiveEpochPool_t) + sizeof(hiveEpoch_t) * poolSize);
    epochPool->index = 0;
    epochPool->outstanding = poolSize;
    epochPool->size = poolSize;
    
    hiveRouteTableAddItem(epochPool, *epochPoolGuid, hiveGlobalRankId, false);
    for(unsigned int i=0; i<poolSize; i++)
    {
        epochPool->pool[i].phase = PHASE_1;
        epochPool->pool[i].poolGuid = *epochPoolGuid;
        epochPool->pool[i].guid = hiveGetGuid(range, i);
        epochPool->pool[i].queued = (hiveIsGuidLocal(*epochPoolGuid)) ? 0 : EpochBit;
        if(!hiveIsGuidLocal(*epochPoolGuid))
        {
            hiveRouteTableAddItemRace(&epochPool->pool[i], epochPool->pool[i].guid, hiveGlobalRankId, false);
            hiveRouteTableFireOO(epochPool->pool[i].guid, hiveOutOfOrderHandler);
        }
    }
    
    DPRINTF("Creating pool %lu starting %lu %p\n", *epochPoolGuid, hiveGetGuid(range, 0), epochPool);
    
    if(newRange)
        hiveFree(range);
    
    return epochPool;
}

void deleteEpoch(hiveGuid_t epochGuid, hiveEpoch_t * epoch)
{
    //Can't call delete unless we already hit two barriers thus it must exit
    if(!epoch)
        epoch = hiveRouteTableLookupItem(epochGuid);
    
    if(epoch->poolGuid)
    {
        hiveEpochPool_t * pool = hiveRouteTableLookupItem(epoch->poolGuid);
        if(hiveIsGuidLocal(epoch->poolGuid))
        {
            hiveRouteTableRemoveItem(epochGuid);
            if(!hiveAtomicSub(&pool->outstanding, 1))
            {
                hiveRouteTableRemoveItem(epoch->poolGuid);
//                hiveFree(pool);  //Free in the next getPoolEpoch
                for(unsigned int i=0; i<hiveGlobalRankCount; i++)
                {
                    if(i!=hiveGlobalRankId)
                        hiveRemoteEpochDelete(i, epochGuid);
                }
            }
        }
        else
        {
            for(unsigned int i=0; i<pool->size; i++)
                hiveRouteTableRemoveItem(pool->pool[i].guid);
            hiveRouteTableRemoveItem(epoch->poolGuid);
            hiveFree(pool);
        }
    }
    else
    {
        hiveRouteTableRemoveItem(epochGuid);
        hiveFree(epoch);
        
        if(hiveIsGuidLocal(epochGuid))
        {
            for(unsigned int i=0; i<hiveGlobalRankCount; i++)
            {
                if(i!=hiveGlobalRankId)
                    hiveRemoteEpochDelete(i, epochGuid);
            }
        }
    }
}

void cleanEpochPool()
{
    DPRINTF("EPOCHTHREADPOOL -------- %p\n", epochThreadPool);
    
    hiveEpochPool_t * trailPool = NULL;
    hiveEpochPool_t * pool = epochThreadPool;
    
    while(pool)
    {
        DPRINTF("###### POOL %p\n", pool);
        if(pool->index == epochThreadPool->size && !pool->outstanding)
        {
            hiveEpochPool_t * toFree = pool;
            DPRINTF("Deleting %p\n", toFree);
            
            pool = pool->next;
            
            if(trailPool)
                trailPool->next = pool;
            else
                epochThreadPool = pool;
            
            hiveFree(toFree);
            DPRINTF("JUST FREED A POOL\n");
        }
        else
        {
            DPRINTF("Next...\n");
            trailPool = pool;
            pool = pool->next;
        }
    }
}

hiveEpoch_t * getPoolEpoch(hiveGuid_t edtGuid, unsigned int slot)
{
    DPRINTF("EpochThreadPool %p\n", epochThreadPool);
    
//    cleanEpochPool();
    hiveEpochPool_t * trailPool = NULL;
    hiveEpochPool_t * pool = epochThreadPool;
    hiveEpoch_t * epoch = NULL;
    while(!epoch)
    {
        if(!pool)
        {
            hiveGuid_t poolGuid = NULL_GUID;
            hiveGuid_t startGuid = NULL_GUID;
            pool = createEpochPool(&poolGuid, DEFAULT_EPOCH_POOL_SIZE, &startGuid);
            
            if(trailPool)
                trailPool->next = pool;
            else
                epochThreadPool = pool;
            
            
            for(unsigned int i=0; i<hiveGlobalRankCount; i++)
            {
                if(i!=hiveGlobalRankId)
                    hiveRemoteEpochInitPoolSend(i, DEFAULT_EPOCH_POOL_SIZE, startGuid, poolGuid);
            }
        }
        
        DPRINTF("Pool index: %u\n", pool->index);
        if(pool->index < pool->size)
            epoch = &pool->pool[pool->index++];
        else
        {
            trailPool = pool;
            pool = pool->next;
        }
    }
    DPRINTF("GetPoolEpoch %lu\n", epoch->guid);
    
    epoch->terminationExitGuid = edtGuid;
    epoch->terminationExitSlot = slot;
    hiveRouteTableAddItemRace(epoch, epoch->guid, hiveGlobalRankId, false);
    hiveRouteTableFireOO(epoch->guid, hiveOutOfOrderHandler);
    return epoch;
}

void hiveYield()
{
    HIVECOUNTERINCREMENT(yield);
    threadLocal_t tl;
    hiveSaveThreadLocal(&tl);
    hiveNodeInfo.scheduler();
    hiveRestoreThreadLocal(&tl);
}

bool hiveWaitOnHandle(hiveGuid_t epochGuid)
{
    hiveGuid_t * guid = hiveCheckEpochIsRoot(epochGuid);
    if(guid)
    {
        hiveGuid_t local = *guid;
        *guid = NULL_GUID; //Unset
        unsigned int flag = 1;
        hiveEpoch_t * epoch = hiveRouteTableLookupItem(local);
        epoch->waitPtr = &flag;
        incrementFinishedEpoch(local);
//        globalShutdownGuidIncFinished();
        
        HIVECOUNTERINCREMENT(yield);
        threadLocal_t tl;
        hiveSaveThreadLocal(&tl);
        while(flag)
            hiveNodeInfo.scheduler();
        hiveRestoreThreadLocal(&tl);
        
        cleanEpochPool();
        
        return true;
    }
    return false;
}