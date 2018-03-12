#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
//#include <stdatomic.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"
#include "hiveAtomics.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveGlobals.h"
#include "hiveRemoteFunctions.h"
#include "hiveRouteTable.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

#define EpochMask   0x7FFFFFFFFFFFFFFF  
#define EpochBit 0x8000000000000000

hiveEpoch_t * theEpoch = NULL;

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
            DPRINTF("OOActive\n");
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
        DPRINTF("OOActive\n");
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
                checkEpoch(epoch, epoch->activeCount, epoch->finishedCount);
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
//                        else
//                            DPRINTF("OUTSTANDING: %lu\n", epoch->outstanding);
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
    epoch->queued = (hiveIsGuidLocal(*guid)) ? 0 : EpochBit;
    hiveRouteTableAddItemRace(epoch, *guid, hiveGlobalRankId, false);
    hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler);
    theEpoch = epoch;
    return epoch;
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
    if(hiveGetCurrentEpochGuid() == NULL_GUID)
    {
        hiveGuid_t guid = NULL_GUID;
        hiveEpoch_t * epoch = createEpoch(&guid, finishEdtGuid, slot);
        
        if(hiveSetCurrentEpochGuid(guid))
        {
            hiveAtomicAdd(&epoch->activeCount, 1);
            hiveAtomicAddU64(&epoch->queued, 1);
        }
        
        for(unsigned int i=0; i<hiveGlobalRankCount; i++)
        {
            if(i != hiveGlobalRankId)
                hiveRemoteEpochInitSend(i, guid, finishEdtGuid, slot);
        }
        
        DPRINTF("%u : %lu --------> %lu\n", epoch->activeCount, epoch->queued, guid);
        return guid;
    }
    PRINTF("Nested Epoch not supported...\n");
    return NULL_GUID;
}

hiveGuid_t hiveInitializeEpoch(hiveGuid_t startEdtGuid, hiveGuid_t finishEdtGuid, unsigned int slot)
{
    struct hiveEdt * edt = hiveRouteTableLookupItem(startEdtGuid);
    if(edt)
    {
        hiveGuid_t guid = NULL_GUID;
        hiveEpoch_t * epoch = createEpoch(&guid, finishEdtGuid, slot);
        hiveAtomicAdd(&epoch->activeCount, 1);
        
        for(unsigned int i=0; i<hiveGlobalRankCount; i++)
        {
            if(i != hiveGlobalRankId)
                hiveRemoteEpochInitSend(i, guid, finishEdtGuid, slot);
        }
        
        edt->epochGuid = guid;
        return guid;
    }
    PRINTF("Out-of-order add to epoch not implemented yet...\n");
    return NULL_GUID;
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
            DPRINTF("%lu Calling finalization continuation provided by the user %u\n", epoch->guid, totalFinish);
            hiveSignalEdt(epoch->terminationExitGuid, totalFinish, epoch->terminationExitSlot, DB_MODE_SINGLE_VALUE);
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
                hiveSignalEdt(epoch->terminationExitGuid, totalFinish, epoch->terminationExitSlot, DB_MODE_SINGLE_VALUE);
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
            
            DPRINTF("%lu EPOCH QUEUEU: %u\n", epochGuid, epoch->queued);
        }
        DPRINTF("###### %lu\n", epoch->outstanding);
    }
    else
        PRINTF("%lu ERROR: NO EPOCH\n", epochGuid);
}
