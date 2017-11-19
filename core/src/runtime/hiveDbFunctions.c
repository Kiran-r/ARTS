#include "hive.h"
#include "hiveDbFunctions.h"
#include "hiveAtomics.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveGlobals.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveDbList.h"
#include "hiveDeque.h"
#include "hiveDebug.h"
#include <string.h>
#define DPRINTF( ... )

void hiveDbCreateInternal(hiveGuid_t guid, void *addr, u64 size, u64 packetSize)
{
    struct hiveHeader *header = (struct hiveHeader*)addr;
    header->type = HIVE_DB;
    header->size = packetSize;

    struct hiveDb * dbRes = (struct hiveDb *)header;
    dbRes->guid = guid;
    dbRes->dbList = hiveNewDbList();
}

hiveGuid_t hiveDbCreate(void **addr, u64 size)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    hiveGuid_t guid = NULL_GUID;
    unsigned int dbSize = size + sizeof(struct hiveDb);

    HIVESETMEMSHOTTYPE(hiveDbMemorySize);
    void * ptr = hiveMalloc(dbSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    if(ptr)
    {
        guid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_DB);
        hiveDbCreateInternal(guid, ptr, size, dbSize);
        //change false to true to force a manual DB delete
        hiveRouteTableAddItem(ptr, guid, hiveGlobalRankId, false);
        *addr = (void*)((struct hiveDb *) ptr + 1);
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return guid;
}

void * hiveDbCreateWithGuid(hiveGuid_t guid, u64 size)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    unsigned int dbSize = size + sizeof(struct hiveDb);

    HIVESETMEMSHOTTYPE(hiveDbMemorySize);
    void * ptr = hiveMalloc(dbSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    if(ptr)
    {
        unsigned int route = hiveGuidGetRank(guid);
        hiveDbCreateInternal(guid, ptr, size, dbSize);
        if(route == hiveGlobalRankId)
        {
            if(hiveRouteTableAddItemRace(ptr, guid, hiveGlobalRankId, false))
                hiveRouteTableFireOO(guid, hiveOutOfOrderHandler);
        }
        else
        {
            //add code to move db here...
        }
        ptr = (void*)((struct hiveDb *) ptr + 1);
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return ptr;
}

void * hiveDbResizePtr(struct hiveDb * dbRes, unsigned int size, bool copy)
{
    if(dbRes)
    {
        unsigned int oldSize = dbRes->header.size;
        unsigned int newSize = size + sizeof(struct hiveDb);
        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
        struct hiveDb *  ptr = hiveCalloc(size + sizeof(struct hiveDb));
        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
        if(ptr)
        {
            if(copy)
                memcpy(ptr, dbRes, oldSize);
            else
                memcpy(ptr, dbRes, sizeof(struct hiveDb));
            hiveFree(dbRes);
            ptr->header.size = size + sizeof(struct hiveDb);
            return (void*)(ptr+1);
        }
    }
    return NULL;
}

//Must be in write mode (or only copy) to update and alloced (no NO_ACQUIRE nonsense), otherwise will be racy...
void * hiveDbResize(hiveGuid_t guid, unsigned int size, bool copy)
{
    struct hiveDb * dbRes = hiveRouteTableLookupItem(guid);
    void * ptr = hiveDbResizePtr(dbRes, size, copy);
    if(ptr)
    {
        dbRes = ((struct hiveDb *)ptr) - 1;
//        hiveRouteTableUpdateItem(dbRes, guid, hiveGlobalRankId);
    }
    return ptr;
}

void hiveDbDestroy(hiveGuid_t guid)
{
    struct hiveDb * dbRes = hiveRouteTableLookupItem(guid);
    if(dbRes!=NULL)
    {
        hiveRemoteDbDestroy(guid, hiveGlobalRankId, 0);
        hiveFree(dbRes);
        hiveRouteTableRemoveItem(guid);
    }
    else
        hiveRemoteDbDestroy(guid, hiveGlobalRankId, 0);
}

void hiveDbDestroySafe(hiveGuid_t guid, bool remote)
{
    struct hiveDb * dbRes = hiveRouteTableLookupItem(guid);
    if(dbRes!=NULL)
    {
        if(remote)
            hiveRemoteDbDestroy(guid, hiveGlobalRankId, 0);
        hiveFree(dbRes);
        hiveRouteTableRemoveItem(guid);
    }
    else if(remote)
        hiveRemoteDbDestroy(guid, hiveGlobalRankId, 0);
}

hiveGuid_t hiveDbAssignGuid(void * ptr)
{
    struct hiveDb * db = ((struct hiveDb *)ptr) - 1;
    hiveGuid_t guid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_DB);
    hiveRouteTableAddItem(db, guid, hiveGlobalRankId, false);
    return guid;
}

hiveGuid_t hiveDbAssignGuidForRank(void * ptr, unsigned int rank)
{
    struct hiveDb * db = ((struct hiveDb *)ptr) - 1;
    hiveGuid_t guid = hiveGuidCreateForRank(rank, HIVE_DB);
    db->guid = guid;
    hiveRouteTableAddItem(db, guid, hiveGlobalRankId, false);
    return guid;
}

hiveGuid_t hiveDbReassignGuid(hiveGuid_t guid)
{
    hiveGuid_t newGuid = NULL_GUID;
    if(guid != NULL_GUID)
    {
        struct hiveDb * dbRes = hiveRouteTableLookupItem(guid);
        dbRes->guid = guid;
//        hiveRouteTableUpdateItem(NULL, guid, hiveGlobalRankId, 2);
        newGuid = hiveGuidCreateForRank(hiveGlobalRankId, HIVE_DB);
        hiveRouteTableAddItem(dbRes, newGuid, hiveGlobalRankId, false);
    }
    return newGuid;
}

void hiveDbCleanExt(hiveGuid_t guid, bool removeLocal)
{
    hiveRemoteDbDestroy(guid, hiveGlobalRankId, 1);
    if(removeLocal && hiveGlobalRankId != hiveGuidGetRank(guid))
    {
        hiveFree(hiveRouteTableLookupItem(guid));
        hiveRouteTableRemoveItem(guid);
    }
}

//This is not threadsafe... Don't use with remote work stealing
void hiveDbCleanLocalOnlyExt(hiveGuid_t guid)
{
    hiveFree(hiveRouteTableLookupItem(guid));
    hiveRouteTableRemoveItem(guid);
}

/**********************DB MEMORY MODEL*************************************/

//Side Effects:/ edt depcNeeded will be incremented, ptr will be updated,
//  and launches out of order handleReadyEdt
//Returns false on out of order and true otherwise
void acquireDbs(struct hiveEdt * edt)
{
    hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
    edt->depcNeeded = edt->depc + 1;

    for(int i=0; i<edt->depc; i++)
    {
        if(depv[i].guid && depv[i].ptr == NULL)
        {
            struct hiveDb * dbFound = NULL;
            int owner = hiveGuidGetRank(depv[i].guid);
            switch(depv[i].mode)
            {
                case DB_MODE_NON_COHERENT_READ:
                case DB_MODE_CDAG_WRITE:
                    if(owner == hiveGlobalRankId) //Owner Rank
                    {
                        int validRank = -1;
                        struct hiveDb * dbTemp = hiveRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp) //We have found an entry
                        {
                            if(hiveAddDbDuplicate(dbTemp, hiveGlobalRankId, edt, i, depv[i].mode))
                            {
                                if(validRank == hiveGlobalRankId) //Owner rank and we have the valid copy
                                {
                                    DPRINTF("%lu Locally resolved\n", depv[i].guid);
                                    dbFound = dbTemp;
                                    hiveAtomicSub(&edt->depcNeeded, 1U);
                                }
                                else //Owner rank but someone else has valid copy
                                {
                                    DPRINTF("%lu Foward to current owner\n", depv[i].guid);
                                    if(depv[i].mode == DB_MODE_NON_COHERENT_READ)
                                        hiveRemoteDbRequest(depv[i].guid, validRank, edt, i, depv[i].mode, true);
                                    else
                                        hiveRemoteDbFullRequest(depv[i].guid, validRank, edt, i, depv[i].mode);
                                }
                            }
                            else  //We can't read right now due to an exclusive access or cdag write in progress
                            {
                                DPRINTF("%lu Queue in frontier\n", depv[i].guid);
                            }
                        }
                        else //The Db hasn't been created yet
                        {
                            DPRINTF("%lu out of order request\n", depv[i].guid);
                            hiveOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        }
                    }
                    else
                    {
                        int validRank = -1;
                        struct hiveDb * dbTemp = hiveRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp) //We have found an entry
                        {
                            DPRINTF("%lu found a valid local copy\n", depv[i].guid);
                            dbFound = dbTemp;
                            hiveAtomicSub(&edt->depcNeeded, 1U);
                        }

                        if(depv[i].mode == DB_MODE_CDAG_WRITE)
                        {
                            //We can't aggregate read requests for cdag write
                            DPRINTF("%lu requesting a copy in write mode\n", depv[i].guid);
                            hiveRemoteDbFullRequest(depv[i].guid, owner, edt, i, depv[i].mode);
                        }
                        else if(!dbTemp)
                        {
                            //We can aggregate read requests for reads
                            DPRINTF("%lu requesting a copy in read mode\n", depv[i].guid);
                            hiveRemoteDbRequest(depv[i].guid, owner, edt, i, depv[i].mode, true);
                        }
                    }
                    break;

                case DB_MODE_NON_COHERENT_WRITE:
                case DB_MODE_SINGLE_VALUE:
                default:
                    hiveAtomicSub(&edt->depcNeeded, 1U);
                    break;
            }

            if(dbFound)
            {
                depv[i].ptr = dbFound + 1;
            }
        }
        else
        {
            hiveAtomicSub(&edt->depcNeeded, 1U);
        }
    }
}

void prepDbs(unsigned int depc, hiveEdtDep_t * depv)
{
    for(unsigned int i=0; i<depc; i++)
    {
        if(   depv[i].guid != NULL_GUID &&
            ( depv[i].mode == DB_MODE_CDAG_WRITE ||
              depv[i].mode == DB_MODE_EXCLUSIVE_WRITE) )
        {
            hiveRemoteUpdateRouteTable(depv[i].guid, -1);
        }
    }
}

void releaseDbs(unsigned int depc, hiveEdtDep_t * depv)
{
    for(int i=0; i<depc; i++)
    {
        if(   depv[i].guid != NULL_GUID &&
            ( depv[i].mode == DB_MODE_CDAG_WRITE ||
              depv[i].mode == DB_MODE_EXCLUSIVE_WRITE ))
        {
            //Signal we finished and progress frontier
            if(hiveGuidGetRank(depv[i].guid) == hiveGlobalRankId)
            {
                struct hiveDb * db = ((struct hiveDb *)depv[i].ptr - 1);
                hiveProgressFrontier(db, hiveGlobalRankId);

            }
            else
            {
                hiveRemoteUpdateDb(depv[i].guid, false);
            }
//            hiveRouteTableReturnDb(depv[i].guid, false);
        }
        else 
        {
            if(hiveRouteTableReturnDb(depv[i].guid, true))
                PRINTF("FREED A COPY!\n");
        }
    }
}

bool hiveAddDbDuplicate(struct hiveDb * db, unsigned int rank, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
    bool write = false;
    bool exclusive = false;
    switch(mode)
    {
        case DB_MODE_EXCLUSIVE_READ:
        case DB_MODE_EXCLUSIVE_WRITE:
            exclusive = true;
        case DB_MODE_CDAG_WRITE:
            write = true;
        case DB_MODE_NON_COHERENT_READ:
        case DB_MODE_NON_COHERENT_WRITE:
        case DB_MODE_SINGLE_VALUE:
        default:
            break;
    }
    return hivePushDbToList(db->dbList, rank, write, exclusive, hiveGuidGetRank(db->guid) == rank, false, edt, slot, mode);
}
