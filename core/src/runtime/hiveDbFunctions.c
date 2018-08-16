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
#include "hiveEdtFunctions.h"
#include "hiveTerminationDetection.h"
#include <string.h>
#define DPRINTF( ... )

void hiveDbCreateInternal(hiveGuid_t guid, void *addr, u64 size, u64 packetSize, hiveType_t mode)
{
    struct hiveHeader *header = (struct hiveHeader*)addr;
    header->type = mode;
    header->size = packetSize;

    struct hiveDb * dbRes = (struct hiveDb *)header;
    dbRes->guid = guid;
    if(mode != HIVE_DB_PIN)
    {
        dbRes->dbList = hiveNewDbList();
    }
}

hiveGuid_t hiveDbCreateRemote(unsigned int route, u64 size, hiveType_t mode)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    hiveGuid_t guid = hiveGuidCreateForRank(route, mode);
    void * ptr = hiveMalloc(sizeof(struct hiveDb));
    struct hiveDb * db = (struct hiveDb*) ptr;
    db->header.size = size + sizeof(struct hiveDb);
    db->dbList = (mode == HIVE_DB_PIN) ? (void*)0 : (void*)1;
    
    hiveRemoteMemoryMove(route, guid, ptr, sizeof(struct hiveDb), HIVE_REMOTE_DB_SEND_MSG, hiveFree);
    HIVEEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
}

//Creates a local DB only
hiveGuid_t hiveDbCreate(void **addr, u64 size, hiveType_t mode)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    hiveGuid_t guid = NULL_GUID;
    unsigned int dbSize = size + sizeof(struct hiveDb);

    HIVESETMEMSHOTTYPE(hiveDbMemorySize);
    void * ptr = hiveMalloc(dbSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    if(ptr)
    {
        guid = hiveGuidCreateForRank(hiveGlobalRankId, mode);
        hiveDbCreateInternal(guid, ptr, size, dbSize, mode);
        //change false to true to force a manual DB delete
        hiveRouteTableAddItem(ptr, guid, hiveGlobalRankId, false);
        *addr = (void*)((struct hiveDb *) ptr + 1);
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return guid;
}

//Guid must be for a local DB only
void * hiveDbCreateWithGuid(hiveGuid_t guid, u64 size)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    hiveType_t mode = hiveGuidGetType(guid);
    void * ptr = NULL;
    if(hiveIsGuidLocal(guid))
    {
        unsigned int dbSize = size + sizeof(struct hiveDb);
        
        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
        ptr = hiveMalloc(dbSize);
        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
        if(ptr)
        {
            unsigned int route = hiveGuidGetRank(guid);
            hiveDbCreateInternal(guid, ptr, size, dbSize, mode);
            if(hiveRouteTableAddItemRace(ptr, guid, hiveGlobalRankId, false))
                hiveRouteTableFireOO(guid, hiveOutOfOrderHandler);
            ptr = (void*)((struct hiveDb *) ptr + 1);
        }
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(dbCreateCounter);
    return ptr;
}

void * hiveDbCreateWithGuidAndData(hiveGuid_t guid, void * data, u64 size)
{
    HIVEEDTCOUNTERTIMERSTART(dbCreateCounter);
    hiveType_t mode = hiveGuidGetType(guid);
    void * ptr = NULL;
    if(hiveIsGuidLocal(guid))
    {
        unsigned int dbSize = size + sizeof(struct hiveDb);
        
        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
        ptr = hiveMalloc(dbSize);
        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
        
        if(ptr)
        {
            unsigned int route = hiveGuidGetRank(guid);
            hiveDbCreateInternal(guid, ptr, size, dbSize, mode);
            void * dbData = (void*)((struct hiveDb *) ptr + 1);
            memcpy(dbData, data, size);
            if(hiveRouteTableAddItemRace(ptr, guid, hiveGlobalRankId, false))
                hiveRouteTableFireOO(guid, hiveOutOfOrderHandler);
            ptr = dbData;
        }
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

void hiveDbMove(hiveGuid_t dbGuid, unsigned int rank)
{
    unsigned int guidRank = hiveGuidGetRank(dbGuid);
    if(guidRank != rank)
    {
        if(guidRank != hiveGlobalRankId)
            hiveDbMoveRequest(dbGuid, rank);
        else
        {
            struct hiveDb * dbRes = hiveRouteTableLookupItem(dbGuid);
            if(dbRes)
                hiveRemoteMemoryMove(rank, dbGuid, dbRes, dbRes->header.size, HIVE_REMOTE_DB_MOVE_MSG, hiveFree);
            else
                hiveOutOfOrderDbMove(dbGuid, rank);
        }
    }
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

/**********************DB MEMORY MODEL*************************************/

hiveTypeName;

//Side Effects:/ edt depcNeeded will be incremented, ptr will be updated,
//  and launches out of order handleReadyEdt
//Returns false on out of order and true otherwise
void acquireDbs(struct hiveEdt * edt)
{
    hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
    edt->depcNeeded = edt->depc + 1;
    for(int i=0; i<edt->depc; i++)
    {
        DPRINTF("MODE: %s\n", getTypeName(depv[i].mode));
        if(depv[i].guid && depv[i].ptr == NULL)
        {
            struct hiveDb * dbFound = NULL;
            int owner = hiveGuidGetRank(depv[i].guid);
            switch(depv[i].mode)
            {
                //This case assumes that the guid exists only on the owner
                case HIVE_DB_ONCE:
                {
                    if(owner != hiveGlobalRankId)
                    {
                        hiveOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        hiveDbMove(depv[i].guid, hiveGlobalRankId);
                        break;
                    }
                    //else fall through to the local case :-p
                }
                case HIVE_DB_ONCE_LOCAL:
                {
                    struct hiveDb * dbTemp = hiveRouteTableLookupItem(depv[i].guid);
                    if(dbTemp)
                    {
                        dbFound = dbTemp;
                    hiveAtomicSub(&edt->depcNeeded, 1U);
                    }
                    else
                        hiveOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                    break;
                }
                case HIVE_DB_PIN:
                {
//                    if(hiveIsGuidLocal(depv[i].guid))
//                    {
                        int validRank = -1;
                        struct hiveDb * dbTemp = hiveRouteTableLookupDb(depv[i].guid, &validRank);
                        if(dbTemp)
                        {
                            dbFound = dbTemp;
                            hiveAtomicSub(&edt->depcNeeded, 1U);
                        }
                        else
                        {
                            hiveOutOfOrderHandleDbRequest(depv[i].guid, edt, i);
                        }
//                    }
//                    else
//                    {
//                        PRINTF("Cannot acquire DB %lu because it is pinned on %u\n", depv[i].guid, hiveGuidGetRank(depv[i].guid));
//                        depv[i].ptr = NULL;
//                        hiveAtomicSub(&edt->depcNeeded, 1U);
//                    }
                    break;
                }
                case HIVE_DB_READ:
                case HIVE_DB_WRITE:
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
                                dbFound = dbTemp;
                                hiveAtomicSub(&edt->depcNeeded, 1U);
                            }
                            else //Owner rank but someone else has valid copy
                            {
                                if(depv[i].mode == HIVE_DB_READ)
                                    hiveRemoteDbRequest(depv[i].guid, validRank, edt, i, depv[i].mode, true);
                                else
                                    hiveRemoteDbFullRequest(depv[i].guid, validRank, edt, i, depv[i].mode);
                            }
                        }
//                            else  //We can't read right now due to an exclusive access or cdag write in progress
//                            {
//                                PRINTF("############### %lu Queue in frontier\n", depv[i].guid);
//                            }
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
                            dbFound = dbTemp;
                            hiveAtomicSub(&edt->depcNeeded, 1U);
                        }

                        if(depv[i].mode == HIVE_DB_WRITE)
                        {
                            //We can't aggregate read requests for cdag write
                            hiveRemoteDbFullRequest(depv[i].guid, owner, edt, i, depv[i].mode);
                        }
                        else if(!dbTemp)
                        {
                            //We can aggregate read requests for reads
                            hiveRemoteDbRequest(depv[i].guid, owner, edt, i, depv[i].mode, true);
                        }
                    }
                    break;

                case HIVE_NULL:
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
              depv[i].mode == HIVE_DB_WRITE )
        {
            hiveRemoteUpdateRouteTable(depv[i].guid, -1);
        }
    }
}

void releaseDbs(unsigned int depc, hiveEdtDep_t * depv)
{
    for(int i=0; i<depc; i++)
    {
//        PRINTF(">>>>>>>>>>>>>>>>>>>>>>> %lu\n", depv[i].guid);
        if(   depv[i].guid != NULL_GUID &&
              depv[i].mode == HIVE_DB_WRITE )
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
        else if(depv[i].mode == HIVE_DB_ONCE_LOCAL || depv[i].mode == HIVE_DB_ONCE)
        {
            hiveRouteTableInvalidateItem(depv[i].guid);
        }
        else if(depv[i].mode == HIVE_PTR)
        {
            hiveFree(depv[i].ptr);
        }
        else 
        {
            if(hiveRouteTableReturnDb(depv[i].guid, depv[i].mode != HIVE_DB_PIN))
                PRINTF("FREED A COPY!\n");
        }
    }
}

bool hiveAddDbDuplicate(struct hiveDb * db, unsigned int rank, struct hiveEdt * edt, unsigned int slot, hiveType_t mode)
{
    bool write = (mode==HIVE_DB_WRITE);
    bool exclusive = false;
    return hivePushDbToList(db->dbList, rank, write, exclusive, hiveGuidGetRank(db->guid) == rank, false, edt, slot, mode);
}

void internalGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    if(rank==hiveGlobalRankId)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        if(db)
        {
            void * data = (void*)(((char*) (db+1)) + offset);
            void * ptr = hiveMalloc(size);
            memcpy(ptr, data, size);
            DPRINTF("GETTING: %u From: %p\n", *(unsigned int*)ptr, data);
            hiveSignalEdtPtr(edtGuid, slot, ptr, size);
            hiveUpdatePerformanceMetric(hiveGetBW, hiveThread, size, false);
        }
        else
        {
            hiveOutOfOrderGetFromDb(edtGuid, dbGuid, slot, offset, size);
        }
    }
    else
    {
        DPRINTF("Sending to %u\n", rank);
        hiveRemoteGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    }
}

void hiveGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    HIVEEDTCOUNTERTIMERSTART(getDbCounter);
    unsigned int rank = hiveGuidGetRank(dbGuid);
    internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    HIVEEDTCOUNTERTIMERENDINCREMENT(getDbCounter);
}

void hiveGetFromDbAt(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    HIVEEDTCOUNTERTIMERSTART(getDbCounter);
    internalGetFromDb(edtGuid, dbGuid, slot, offset, size, rank);
    HIVEEDTCOUNTERTIMERENDINCREMENT(getDbCounter);
}

void internalPutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epochGuid, unsigned int rank)
{
    if(rank==hiveGlobalRankId)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        if(db)
        {
            //Do this so when we increment finished we can check the term status
            incrementQueueEpoch(epochGuid);
            globalShutdownGuidIncQueue();
            void * data = (void*)(((char*) (db+1)) + offset);
            memcpy(data, ptr, size);
            DPRINTF("PUTTING %u From: %p\n", *((unsigned int *)data), data);
            if(edtGuid)
            {
                hiveSignalEdt(edtGuid, slot, dbGuid);
            }
            DPRINTF("FINISHING PUT %lu\n", epochGuid);
            incrementFinishedEpoch(epochGuid);
            globalShutdownGuidIncFinished();
            hiveUpdatePerformanceMetric(hivePutBW, hiveThread, size, false);
        }
        else
        {
            void * cpyPtr = hiveMalloc(size);
            memcpy(cpyPtr, ptr, size);
            hiveOutOfOrderPutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size, epochGuid);
        }
    }
    else
    {
        void * cpyPtr = hiveMalloc(size);
        memcpy(cpyPtr, ptr, size);
        hiveRemotePutInDb(cpyPtr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    }
}

void hivePutInDbAt(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank)
{
    HIVEEDTCOUNTERTIMERSTART(putDbCounter);
    hiveGuid_t epochGuid = hiveGetCurrentEpochGuid();
    DPRINTF("EPOCH %lu\n", epochGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    HIVEEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}

void hivePutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    HIVEEDTCOUNTERTIMERSTART(putDbCounter);
    unsigned int rank = hiveGuidGetRank(dbGuid);
    hiveGuid_t epochGuid = hiveGetCurrentEpochGuid();
    DPRINTF("EPOCH %lu\n", epochGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    HIVEEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}

void hivePutInDbEpoch(void * ptr, hiveGuid_t epochGuid, hiveGuid_t dbGuid, unsigned int offset, unsigned int size)
{
    HIVEEDTCOUNTERTIMERSTART(putDbCounter);
    unsigned int rank = hiveGuidGetRank(dbGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, NULL_GUID, dbGuid, 0, offset, size, epochGuid, rank);
    HIVEEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}
