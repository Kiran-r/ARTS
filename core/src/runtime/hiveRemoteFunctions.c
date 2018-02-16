#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGlobals.h"
#include "hiveGuid.h"
#include "hiveRemoteProtocol.h"
#include "hiveRouteTable.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveRuntime.h"
#include "hiveOutOfOrder.h"
#include "hiveAtomics.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveDbList.h"
#include "hiveDebug.h"
#include "hiveEdtFunctions.h"
#include "hiveServer.h"
#include "hiveQueue.h"
#include "hiveTerminationDetection.h"
#include "hiveArrayDb.h"
#include <unistd.h>

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

struct hiveInvalidatePingBack
{
    unsigned int awaiting;
    unsigned int signalRank;
    volatile void * volatile signalMe;
};

static inline void hiveFillPacketHeader(struct hiveRemotePacket * header, unsigned int size, unsigned int messageType)
{
    header->size = size;
    header->messageType = messageType;
    header->rank = hiveGlobalRankId;
}

void hiveRemoteAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode, unsigned int rank)
{
    DPRINTF("Remote Add dependence sent %d\n", rank);
    struct hiveRemoteAddDependencePacket packet;
    packet.source = source;
    packet.destination = destination;
    packet.slot = slot;
    packet.mode = mode;
    packet.destRoute = hiveGuidGetRank(destination); 
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_ADD_DEPENDENCE_MSG);
    hiveRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
}

void hiveRemoteUpdateRouteTable(hiveGuid_t guid, unsigned int rank)
{
    DPRINTF("Here Update Table %ld %u\n", guid, rank);
    unsigned int owner = hiveGuidGetRank(guid);
    if(owner == hiveGlobalRankId)
    {
        struct hiveDbFrontierIterator * iter = hiveRouteTableGetRankDuplicates(guid, rank);
        if(iter)
        {
            unsigned int node;
            while(hiveDbFrontierIterNext(iter, &node))
            {
                if(node != hiveGlobalRankId && node != rank)
                {
                    struct hiveRemoteInvalidateDbPacket outPacket;
                    outPacket.guid = guid;
                    hiveFillPacketHeader(&outPacket.header, sizeof(outPacket), HIVE_REMOTE_INVALIDATE_DB_MSG);
                    hiveRemoteSendRequestAsync(node, (char *)&outPacket, sizeof(outPacket));
                }
            }
            hiveFree(iter);
        }
    }
    else
    {         
        struct hiveRemoteUpdateDbGuidPacket packet;
        hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_UPDATE_GUID_MSG);
        packet.guid = guid;
        hiveRemoteSendRequestAsync(owner, (char *)&packet, sizeof(packet));
    }

}

void hiveRemoteHandleUpdateDbGuid(void * ptr)
{
    struct hiveRemoteUpdateDbGuidPacket * packet = ptr;
    DPRINTF("Updated %ld to %d\n", packet->guid, packet->header.rank);
    hiveRemoteUpdateRouteTable(packet->guid, packet->header.rank);
}

void hiveRemoteHandleInvalidateDb(void * ptr)
{
    struct hiveRemoteInvalidateDbPacket * packet = ptr;
    void * address = hiveRouteTableLookupItem(packet->guid);
    hiveRouteTableInvalidateItem(packet->guid);
}

void hiveRemoteDbDestroy(hiveGuid_t guid, unsigned int originRank, bool clean)
{
//    unsigned int rank = hiveGuidGetRank(guid);
//    //PRINTF("Destroy Check\n");
//    if(rank == hiveGlobalRankId)
//    {
//        struct hiveRouteInvalidate * table = hiveRouteTableGetRankDuplicates(guid);
//        struct hiveRouteInvalidate * next = table;
//        struct hiveRouteInvalidate * current;
//        
//        if(next != NULL && next->used != 0)
//        {
//            struct hiveRemoteGuidOnlyPacket outPacket;
//            outPacket.guid = guid;
//            hiveFillPacketHeader(&outPacket.header, sizeof(outPacket), HIVE_REMOTE_DB_DESTROY_MSG);
//            
//            int lastSend=-1;
//            while( next != NULL)
//            {
//                for(int i=0; i < next->used; i++ )
//                {
//                    if(originRank != next->data[i] && next->data[i] != lastSend)
//                    {
////                        PRINTF("Destroy Send 1\n");
//                        lastSend = next->data[i];
//                        hiveRemoteSendRequestAsync(next->data[i], (char *)&outPacket, sizeof(outPacket));
//                    }
//                }
//                next->used = 0;
//                //current=next;
//                next = next->next;
//                //hiveFree(current);
//            }
//        } 
//        if(originRank != hiveGlobalRankId && !clean)
//        {
////            PRINTF("Origin Destroy\n");
////            hiveDebugPrintStack();
//            void * address = hiveRouteTableLookupItem(guid);
//            hiveFree(address);
//            hiveRouteTableRemoveItem(guid);
//        }
//        //if( originRank != hiveGlobalRankId )
//        //    hiveDbDestroy(guid);
//    }
//    else
//    {
//        //void * dbAddress = hiveRouteTableLookupItem(  guid );
//        //DPRINTF("depv %ld %p %p\n", guid, dbAddress, callBack);            
//        struct hiveRemoteGuidOnlyPacket packet;
//        if(!clean)
//            hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_DESTROY_FORWARD_MSG);
//        else 
//            hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_CLEAN_FORWARD_MSG);
//        packet.guid = guid;
////        PRINTF("Destroy Send 2\n");
////        hiveDebugPrintStack();
//        hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
//    }
}

void hiveRemoteHandleDbDestroyForward(void * ptr)
{
    //PRINTF("Destroy\n");
    struct hiveRemoteGuidOnlyPacket * packet = ptr;
    hiveRemoteDbDestroy(packet->guid, packet->header.rank, 0);
    hiveDbDestroySafe(packet->guid, false);
}

void hiveRemoteHandleDbCleanForward(void * ptr)
{
    struct hiveRemoteGuidOnlyPacket * packet = ptr;
    hiveRemoteDbDestroy(packet->guid, packet->header.rank, 1);
}

void hiveRemoteHandleDbDestroy(void * ptr)
{
    struct hiveRemoteGuidOnlyPacket * packet = ptr;
    //PRINTF("Deleted %ld\n", packet->guid);
    //PRINTF("Destroy\n");
    hiveDbDestroySafe(packet->guid, false);
}

void hiveRemoteUpdateDb(hiveGuid_t guid, bool sendDb)
{
//    sendDb = true;
    unsigned int rank = hiveGuidGetRank(guid);
    if(rank != hiveGlobalRankId)
    {
        struct hiveRemoteUpdateDbPacket packet;
        packet.guid = guid;
        struct hiveDb * db = NULL;
        if(sendDb && (db = hiveRouteTableLookupItem(guid)))
        {
            int size = sizeof(struct hiveRemoteUpdateDbPacket)+db->header.size;
            hiveFillPacketHeader(&packet.header, size, HIVE_REMOTE_DB_UPDATE_MSG);
            hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
        }
        else
        {
            hiveFillPacketHeader(&packet.header, sizeof(struct hiveRemoteUpdateDbPacket), HIVE_REMOTE_DB_UPDATE_MSG);
            hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
        }
    }
}

void hiveRemoteHandleUpdateDb(void * ptr)
{
    struct hiveRemoteUpdateDbPacket * packet = (struct hiveRemoteUpdateDbPacket *) ptr;
    struct hiveDb * packetDb = (struct hiveDb *)(packet+1);
    unsigned int rank = hiveGuidGetRank(packet->guid);
    if(rank == hiveGlobalRankId)
    {
        struct hiveDb ** dataPtr;
        bool write = packet->header.size > sizeof(struct hiveRemoteUpdateDbPacket);
        itemState state = hiveRouteTableLookupItemWithState(packet->guid, (void***)&dataPtr, allocatedKey, write);
        struct hiveDb * db = (dataPtr) ? *dataPtr : NULL;
//        PRINTF("DB: %p %lu %u %u %d\n", db, packet->guid, packet->header.size, sizeof(struct hiveRemoteUpdateDbPacket), state);
        if(write)
        {
            void * ptr = (void*)(db+1);
            memcpy(ptr, packetDb, db->header.size - sizeof(struct hiveDb));
            hiveRouteTableSetRank(packet->guid, hiveGlobalRankId);
            hiveProgressFrontier(db, hiveGlobalRankId);
            hiveRouteTableDecItem(packet->guid, dataPtr);
        }
        else
        {
            hiveProgressFrontier(db, packet->header.rank);
        }
    }
}

void hiveRemoteMemoryMove(unsigned int route, hiveGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*))
{
    struct hiveRemoteMemoryMovePacket packet;
    hiveFillPacketHeader(&packet.header, sizeof(packet)+memSize, messageType);
    packet.guid = guid;
    hiveRemoteSendRequestPayloadAsyncFree(route, (char *)&packet, sizeof(packet), ptr, 0, memSize, guid, freeMethod);
    hiveRouteTableRemoveItem(guid);
}

void hiveRemoteHandleEdtMove(void * ptr)
{
    struct hiveRemoteMemoryMovePacket * packet = ptr ;    
    unsigned int size = packet->header.size - sizeof(struct hiveRemoteMemoryMovePacket);

    HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
    struct hiveEdt * edt = hiveMalloc(size);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);

    memcpy(edt, packet+1, size);
    hiveRouteTableAddItemRace(edt, (hiveGuid_t) packet->guid, hiveGlobalRankId, false);
    if(edt->depcNeeded == 0) 
        hiveHandleReadyEdt(edt);
    else   
        hiveRouteTableFireOO(packet->guid, hiveOutOfOrderHandler);           
}

void hiveRemoteHandleDbMove(void * ptr)
{   
    struct hiveRemoteMemoryMovePacket * packet = ptr ;
    unsigned int size = packet->header.size - sizeof(struct hiveRemoteMemoryMovePacket);
    
    struct hiveDb * dbHeader = (struct hiveDb *)(packet+1);
    unsigned int dbSize  = dbHeader->header.size;
    
    HIVESETMEMSHOTTYPE(hiveDbMemorySize);
    struct hiveHeader * memPacket = hiveMalloc(dbSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    
    if(size == dbSize)
        memcpy(memPacket, packet+1, size);
    else
    {
        memPacket->type = HIVE_DB;
        memPacket->size = dbSize;
    }
    //We need a local pointer for this node
    if(dbHeader->dbList)
    {
        struct hiveDb * newDb = (struct hiveDb*)memPacket;
        newDb->dbList = hiveNewDbList();
    }
    
    if(hiveRouteTableAddItemRace(memPacket, (hiveGuid_t) packet->guid, hiveGlobalRankId, false))
        hiveRouteTableFireOO(packet->guid, hiveOutOfOrderHandler);
}

void hiveRemoteHandleEventMove(void * ptr)
{
    struct hiveRemoteMemoryMovePacket * packet = ptr ;
    unsigned int size = packet->header.size - sizeof(struct hiveRemoteMemoryMovePacket);
    
    HIVESETMEMSHOTTYPE(hiveEventMemorySize);
    struct hiveHeader * memPacket = hiveMalloc(size);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    
    memcpy(memPacket, packet+1, size);
    hiveRouteTableAddItemRace(memPacket, (hiveGuid_t) packet->guid, hiveGlobalRankId, false);
    hiveRouteTableFireOO(packet->guid, hiveOutOfOrderHandler);
}

void hiveRemoteSignalEdt(hiveGuid_t edt, hiveGuid_t db, u32 slot, hiveDbAccessMode_t mode)
{
    DPRINTF("Remote Signal %ld %ld %d %d\n",edt,db,slot, hiveGuidGetRank(edt));
    struct hiveRemoteEdtSignalPacket packet;
    
    unsigned int rank = hiveGuidGetRank(edt); 

    if(rank == hiveGlobalRankId)
        rank = hiveRouteTableLookupRank( edt);
    packet.db = db;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    packet.dbRoute = hiveGuidGetRank(db); 
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_EDT_SIGNAL_MSG);
    hiveRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
//    if(hiveGlobalRankId==1) {PRINTF("Size: %u\n", sizeof(packet)); hiveDebugPrintStack();}
}

void hiveRemoteSendStealRequest( unsigned int rank)
{
    static int whom=0;
    struct hiveRemotePacket stealPacket;

    stealPacket.rank=hiveGlobalRankId;
    stealPacket.messageType = HIVE_REMOTE_EDT_STEAL_MSG;
    stealPacket.size = sizeof(stealPacket);
    DPRINTF("Steal %d %d\n", rank, stealPacket.rank);
    hiveRemoteSendRequestAsync(rank, (char *)&stealPacket, sizeof(stealPacket) );
}

void hiveRemoteEventSatisfy(hiveGuid_t eventGuid, hiveGuid_t dataGuid )
{
    DPRINTF("Remote Satisfy sent %ld %ld\n", eventGuid, dataGuid);
    struct hiveRemoteEventSatisfyPacket packet;
    packet.event = eventGuid;
    packet.db = dataGuid;
    packet.dbRoute = hiveGuidGetRank(dataGuid); 
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_EVENT_SATISFY_MSG);
    hiveRemoteSendRequestAsync(hiveGuidGetRank( eventGuid ), (char *)&packet, sizeof(packet) );
}

void hiveRemoteEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot )
{  
    DPRINTF("Remote Satisfy Slot\n");
    struct hiveRemoteEventSatisfySlotPacket packet;
    packet.event = eventGuid;
    packet.db = dataGuid;
    packet.slot = slot;
    packet.dbRoute = hiveGuidGetRank(dataGuid); 
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_EVENT_SATISFY_SLOT_MSG);
    hiveRemoteSendRequestAsync(hiveGuidGetRank( eventGuid ), (char *)&packet, sizeof(packet) );
}

void hiveDbRequestCallback(struct hiveEdt *edt, unsigned int slot, struct hiveDb * dbRes)
{ 
    hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
    depv[slot].ptr = dbRes + 1;
    unsigned int temp = hiveAtomicSub(&edt->depcNeeded, 1U);
    if(temp == 0)
        hiveHandleRemoteStolenEdt(edt);
}

bool hiveRemoteDbRequest(hiveGuid_t dataGuid, int rank, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode, bool aggRequest)
{
    if(hiveRouteTableAddSent(dataGuid, edt, pos, aggRequest))
    {
        struct hiveRemoteDbRequestPacket packet;
        packet.dbGuid = dataGuid;
        packet.mode = mode;
        hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_REQUEST_MSG);
        hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
        DPRINTF("DB req send: %u -> %u mode: %u agg: %u\n", packet.header.rank, rank, mode, aggRequest);
        return true;
    }
    return false;
}

void hiveRemoteDbForward(int destRank, int sourceRank, hiveGuid_t dataGuid, hiveDbAccessMode_t mode)
{
    struct hiveRemoteDbRequestPacket packet;
    packet.header.size = sizeof(packet);
    packet.header.messageType = HIVE_REMOTE_DB_REQUEST_MSG;
    packet.header.rank = destRank;
    packet.dbGuid = dataGuid;
    packet.mode = mode;
    hiveRemoteSendRequestAsync(sourceRank, (char *)&packet, sizeof(packet));
}

void hiveRemoteDbSendNow(int rank, struct hiveDb * db)
{
    DPRINTF("SEND NOW: %u -> %u\n", hiveGlobalRankId, rank);
//    hiveDebugPrintStack();
    struct hiveRemoteDbSendPacket packet;
    int size = sizeof(struct hiveRemoteDbSendPacket)+db->header.size;
    hiveFillPacketHeader(&packet.header, size, HIVE_REMOTE_DB_SEND_MSG);
    hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
}

void hiveRemoteDbSendCheck(int rank, struct hiveDb * db, hiveDbAccessMode_t mode)
{
    if(!hiveIsGuidLocal(db->guid))
    {
        hiveRouteTableReturnDb(db->guid, false);
        hiveRemoteDbSendNow(rank, db);
    }
    else if(hiveAddDbDuplicate(db, rank, NULL, 0, mode))
    {
        hiveRemoteDbSendNow(rank, db);
    }
}

void hiveRemoteDbSend(struct hiveRemoteDbRequestPacket * pack)
{
    unsigned int redirected = hiveRouteTableLookupRank(pack->dbGuid);
    if(redirected != hiveGlobalRankId && redirected != -1)
        hiveRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
    else
    {
        struct hiveDb * db = hiveRouteTableLookupItem(pack->dbGuid);
        if(db == NULL)
        {
            hiveOutOfOrderHandleRemoteDbSend(pack->header.rank, pack->dbGuid, pack->mode);
        }
        else if(!hiveIsGuidLocal(db->guid) && pack->header.rank == hiveGlobalRankId)
        {
            //This is when the memory model sends a CDAG write after CDAG write to the same node
            //The hiveIsGuidLocal should be an extra check, maybe not required
            hiveRouteTableFireOO(pack->dbGuid, hiveOutOfOrderHandler);
        }
        else
            hiveRemoteDbSendCheck(pack->header.rank, db, pack->mode);
    }
}

void hiveRemoteHandleDbRecieved(struct hiveRemoteDbSendPacket * packet)
{
    struct hiveDb * packetDb = (struct hiveDb *)(packet+1);    
    struct hiveDb * dbRes = NULL;
    struct hiveDb ** dataPtr = NULL;
    itemState state = hiveRouteTableLookupItemWithState(packetDb->guid, (void***)&dataPtr, allocatedKey, true);
    
    struct hiveDb * tPtr = (dataPtr) ? *dataPtr : NULL;
    struct hiveDbList * dbList = NULL;
    if(tPtr && hiveIsGuidLocal(packetDb->guid))    
        dbList = tPtr->dbList;
    DPRINTF("Rec DB State: %u\n", state);
    switch(state)
    {              
        case requestedKey:
            if(packetDb->header.size == tPtr->header.size)
            {
                void * source = (void*)((struct hiveDb *) packetDb + 1);
                void * dest = (void*)((struct hiveDb *) tPtr + 1);
                memcpy(dest, source, packetDb->header.size - sizeof(struct hiveDb));
                tPtr->dbList = dbList;
                dbRes = tPtr;
            }
            else
            {
                PRINTF("Did the DB do a remote resize...\n");
            }
            break;
            
        case reservedKey:
            HIVESETMEMSHOTTYPE(hiveDbMemorySize);
            dbRes = hiveMalloc(packetDb->header.size);
            HIVESETMEMSHOTTYPE(hiveDbMemorySize);
            memcpy(dbRes, packetDb, packetDb->header.size);
            if(hiveIsGuidLocal(packetDb->guid))
               dbRes->dbList = hiveNewDbList(); 
            else
                dbRes->dbList = NULL;
            break;
            
        default:
            PRINTF("Got a DB but current key state is %d looking again\n", state);
            itemState state = hiveRouteTableLookupItemWithState(packetDb->guid, (void*)&tPtr, anyKey, false);
            PRINTF("The current state after re-checking is %d\n", state);
            break;
    }
    if(dbRes && hiveRouteTableUpdateItem(packetDb->guid, (void*)dbRes, hiveGlobalRankId, state))
    {
        hiveRouteTableFireOO(packetDb->guid, hiveOutOfOrderHandler);
    }
    
    hiveRouteTableDecItem(packetDb->guid, dataPtr);
}

void hiveRemoteDbFullRequest(hiveGuid_t dataGuid, int rank, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode)
{
    //Do not try to reduce full requests since they are unique
    struct hiveRemoteDbFullRequestPacket packet;
    packet.dbGuid = dataGuid;
    packet.edt = edt;
    packet.slot = pos;
    packet.mode = mode;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_FULL_REQUEST_MSG);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteDbForwardFull(int destRank, int sourceRank, hiveGuid_t dataGuid, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode)
{
    struct hiveRemoteDbFullRequestPacket packet;
    packet.header.size = sizeof(packet);
    packet.header.messageType = HIVE_REMOTE_DB_FULL_REQUEST_MSG;
    packet.header.rank = destRank;
    packet.dbGuid = dataGuid;
    packet.edt = edt;
    packet.slot = pos;
    packet.mode = mode;
    hiveRemoteSendRequestAsync(sourceRank, (char *)&packet, sizeof(packet));
}

void hiveRemoteDbFullSendNow(int rank, struct hiveDb * db, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
    PRINTF("SEND FULL NOW: %u -> %u\n", hiveGlobalRankId, rank);
    struct hiveRemoteDbFullSendPacket packet;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    int size = sizeof(struct hiveRemoteDbFullSendPacket)+db->header.size;
    hiveFillPacketHeader(&packet.header, size, HIVE_REMOTE_DB_FULL_SEND_MSG);
    hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)db, db->header.size);
}

void hiveRemoteDbFullSendCheck(int rank, struct hiveDb * db, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
    if(!hiveIsGuidLocal(db->guid))
    {
        hiveRouteTableReturnDb(db->guid, false);
        hiveRemoteDbFullSendNow(rank, db, edt, slot, mode);
    }
    else if(hiveAddDbDuplicate(db, rank, edt, slot, mode))
    {
        hiveRemoteDbFullSendNow(rank, db, edt, slot, mode);
    }
}

void hiveRemoteDbFullSend(struct hiveRemoteDbFullRequestPacket * pack)
{
    unsigned int redirected = hiveRouteTableLookupRank(pack->dbGuid);
    if(redirected != hiveGlobalRankId && redirected != -1)
        hiveRemoteSendRequestAsync(redirected, (char *)pack, pack->header.size);
    else
    {
        struct hiveDb * db = hiveRouteTableLookupItem(pack->dbGuid);
        if(db == NULL)
        {
            hiveOutOfOrderHandleRemoteDbFullSend(pack->header.rank, pack->dbGuid, pack->edt, pack->slot, pack->mode);
        }
        else
            hiveRemoteDbFullSendCheck(pack->header.rank, db, pack->edt, pack->slot, pack->mode);
    }
}

void hiveRemoteHandleDbFullRecieved(struct hiveRemoteDbFullSendPacket * packet)
{
    bool dec;
    itemState state;
    struct hiveDb * packetDb = (struct hiveDb *)(packet+1);    
    void ** dataPtr = hiveRouteTableReserve(packetDb->guid, &dec, &state);
    struct hiveDb * dbRes = (dataPtr) ? *dataPtr : NULL;    
    if(dbRes)
    {
        if(packetDb->header.size == dbRes->header.size)
        {
            struct hiveDbList * dbList = dbRes->dbList;
            void * source = (void*)((struct hiveDb *) packetDb + 1);
            void * dest = (void*)((struct hiveDb *) dbRes + 1);
            memcpy(dest, source, packetDb->header.size - sizeof(struct hiveDb));
            dbRes->dbList = dbList;
        }
        else
            PRINTF("Did the DB do a remote resize...\n");
    }
    else
    {
        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
        dbRes = hiveMalloc(packetDb->header.size);
        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
        memcpy(dbRes, packetDb, packetDb->header.size);
        if(hiveIsGuidLocal(packetDb->guid))
           dbRes->dbList = hiveNewDbList();
        else
            dbRes->dbList = NULL;
    }
    if(hiveRouteTableUpdateItem(packetDb->guid, (void*)dbRes, hiveGlobalRankId, state))
        hiveRouteTableFireOO(packetDb->guid, hiveOutOfOrderHandler);
    hiveDbRequestCallback(packet->edt, packet->slot, dbRes);
    if(dec)
        hiveRouteTableDecItem(packetDb->guid, dataPtr);
}

void hiveRemoteSendAlreadyLocal(int rank, hiveGuid_t guid, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
    struct hiveRemoteDbFullRequestPacket packet;
    packet.dbGuid = guid;
    packet.edt = edt;
    packet.slot = slot;
    packet.mode = mode;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_DB_FULL_SEND_ALREADY_LOCAL);
    hiveRemoteSendRequestAsync(rank, (char*)&packet, sizeof(packet));
}

void hiveRemoteHandleSendAlreadyLocal(void * pack)
{
    struct hiveRemoteDbFullRequestPacket * packet = pack;
    int rank;
    struct hiveDb * dbRes = hiveRouteTableLookupDb(packet->dbGuid, &rank);
    hiveDbRequestCallback(packet->edt, packet->slot, dbRes);
}

void hiveRemoteGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    unsigned int rank = hiveGuidGetRank(dbGuid);
    struct hiveRemoteGetPutPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.slot = slot;
    packet.offset = offset;
    packet.size = size;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_GET_FROM_DB);
    hiveRemoteSendRequestAsync(rank, (char*)&packet, sizeof(packet));
}

void hiveRemoteHandleGetFromDb(void * pack)
{
    struct hiveRemoteGetPutPacket * packet = pack;
    hiveGetFromDb(packet->edtGuid, packet->dbGuid, packet->slot, packet->offset, packet->size);
}

void hiveRemotePutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epochGuid)
{
    unsigned int rank = hiveGuidGetRank(dbGuid);
    struct hiveRemoteGetPutPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.offset = offset;
    packet.size = size;
    int totalSize = sizeof(struct hiveRemoteGetPutPacket)+size;
    hiveFillPacketHeader(&packet.header, totalSize, HIVE_REMOTE_PUT_IN_DB);
//    hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)ptr, size);
    hiveRemoteSendRequestPayloadAsyncFree(rank, (char*)&packet, sizeof(packet), (char *)ptr, 0, size, NULL_GUID, hiveFree);
}

void hiveRemoteHandlePutInDb(void * pack)
{
    struct hiveRemoteGetPutPacket * packet = pack;
    void * data = (void*)(packet+1);
    internalPutInDb(data, packet->edtGuid, packet->dbGuid, packet->slot, packet->offset, packet->size, packet->epochGuid);
}

void hiveRemoteSignalEdtWithPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot)
{
    unsigned int rank = hiveGuidGetRank(edtGuid);
    DPRINTF("SEND NOW: %u -> %u\n", hiveGlobalRankId, rank);
    struct hiveRemoteSignalEdtWithPtrPacket packet;
    packet.edtGuid = edtGuid;
    packet.dbGuid = dbGuid;
    packet.size = size;
    packet.slot = slot;
    int totalSize = sizeof(struct hiveRemoteSignalEdtWithPtrPacket)+size;
    hiveFillPacketHeader(&packet.header, totalSize, HIVE_REMOTE_SIGNAL_EDT_WITH_PTR);
    hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)ptr, size);
}

void hiveRemoteHandleSignalEdtWithPtr(void * pack)
{
    struct hiveRemoteSignalEdtWithPtrPacket * packet = pack;
    void * source = (void*)(packet + 1);
    void * dest = hiveMalloc(packet->size);
    memcpy(dest, source, packet->size);
    hiveSignalEdtPtr(packet->edtGuid, packet->dbGuid, dest, packet->size, packet->slot);
}

//void hiveRemoteHandleDbExclusiveRequest(struct hiveRemoteDbExclusiveRequestPacket * pack)
//{
//    struct hiveDb * db = hiveRouteTableLookupItem(pack->dbGuid);
//    if(db == NULL)
//    {
//        hiveOutOfOrderHandleRemoteDbExclusiveRequest(pack->dbGuid, pack->header.rank, pack->edt, pack->slot, pack->mode);
//        return;
//    }
//    if(hiveAddDbDuplicate(db, pack->header.rank, pack->edt, pack->slot, pack->mode))
//        hiveRemoteDbExclusiveSend(pack->header.rank, db, pack->edt, pack->slot, pack->mode);
//}
//
//void hiveRemoteHandleDbExclusiveRecieved(struct hiveRemoteDbExclusiveSendPacket * packet)
//{
//    struct hiveDb * packetDb = (struct hiveDb *)(packet+1);    
//    struct hiveDb * dbRes = NULL;
//    
//    struct hiveDb * tPtr = NULL;
//    itemState state = hiveRouteTableLookupItemWithState(packetDb->guid, (void*)&tPtr, anyKey);
//    
//    struct hiveDbList * dbList = NULL;
//    if(tPtr && hiveIsGuidLocal(packetDb->guid))    
//        dbList = tPtr->dbList;
//    
//    switch(state)
//    {
//        case deletedKey:
//            PRINTF("Deleted key received not supported yet...");
//        case reservedKey:
//        case allocatedKey:
//            hiveRouteTableInvalidateItem(packetDb->guid);
//        case noKey:
//        default:
//            PRINTF("GOT A REMOTE ONE... %lu\n", packetDb->guid);
//            HIVESETMEMSHOTTYPE(hiveDbMemorySize);
//            dbRes = hiveMalloc(packetDb->header.size);
//            HIVESETMEMSHOTTYPE(hiveDbMemorySize);
//
//            memcpy(dbRes, packetDb, packetDb->header.size);
//            if(!hiveRouteTableAddItemRace((void*)dbRes, packetDb->guid, hiveGlobalRankId, true))
//                PRINTF("HOW DID SOMEONE DO THAT\n");
//            hiveDbRequestCallback(packet->edt, packet->slot, dbRes);       
//    }
//    dbRes->dbList = dbList;
//}

bool hiveRemoteShutdownSend()
{
    struct hiveRemotePacket shutdownPacket;
    shutdownPacket.rank=hiveGlobalRankId;
    shutdownPacket.messageType = HIVE_REMOTE_SHUTDOWN_MSG;
    shutdownPacket.size = sizeof(shutdownPacket);
    return hiveLLServerSyncEndSend( (char *)&shutdownPacket, sizeof(shutdownPacket));
}

unsigned int packageEdt( void * edtPacket, void ** package )
{

    struct hiveHeader * header = edtPacket;

    unsigned int size = sizeof( struct hiveRemotePacket )+header->size;

    struct hiveRemotePacket * packet = hiveMalloc( size );

    packet->messageType = HIVE_REMOTE_EDT_RECV_MSG;
    packet->rank=hiveGlobalRankId;
    packet->size = size;
    memcpy( packet+1, edtPacket, header->size );

    *package = packet;

    return size;
}

unsigned int packageEdts( void ** edtPackets, int edtCount, void ** package )
{

    int i;
    unsigned int size = sizeof( struct hiveRemotePacket );
    struct hiveHeader * header;
    struct hiveEdt *edt;
    char * ptr;

    for(i=0; i<edtCount; i++)
    {
        header = (struct hiveHeader *)edtPackets[i];
        size+=header->size;
    }

    struct hiveRemotePacket * packet = hiveMalloc( size );

    packet->messageType = HIVE_REMOTE_EDT_RECV_MSG;
    packet->rank=hiveGlobalRankId;
    packet->size = size;

    ptr = (char*)(packet+1);
    for(i=0; i<edtCount; i++)
    {
        header = (struct hiveHeader *)edtPackets[i];
        //hiveRouteTableUpdateItem( NULL, (hiveGuid_t) guid, originRank, invalidate);
        memcpy( ptr, header, header->size );
        ptr+=header->size;
    }
    *package = packet;
    return size;
}

unsigned int packageEdtsAndDbs( void ** edtPackets, int edtCount, void ** package, int rank )
{

    int i, j;
    unsigned int size = sizeof( struct hiveRemotePacket );
    unsigned int finalSize = size;
    struct hiveHeader * header;
    struct hiveEdt *edt;
    char * ptr;
    hiveEdtDep_t *depv;
    u32 depc;
    bool res;

    DPRINTF("------------Packaging edts-----------\n");
    for(i=0; i<edtCount; i++)
    {
        header = (struct hiveHeader *)edtPackets[i];

        size+=header->size;
        edt = (struct hiveEdt *)header;
        depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);        
        depc = edt->depc;

        for(j=0; j<depc; j++)
        {
            header = (struct hiveHeader * ) hiveRouteTableLookupItem( (hiveGuid_t) depv[j].guid );
            res = false;

            if( header != NULL && !res )
            {
                size+=header->size;
            }
        }
    }

    struct hiveRemotePacket * packet = hiveMalloc( size );

    packet->messageType = HIVE_REMOTE_EDT_RECV_MSG;
    packet->rank=hiveGlobalRankId;
    packet->size = size;

    ptr = (char*)(packet+1);
    for(i=0; i<edtCount; i++)
    {
        header = (struct hiveHeader *)edtPackets[i];
        memcpy( ptr, header, header->size );
        depv = (hiveEdtDep_t *)(((u64 *)(((struct hiveEdt*) ptr) + 1)) + edt->paramc);
        ptr+=header->size;
        finalSize+=header->size;

        edt = (struct hiveEdt *)header;
        depc = edt->depc;
        DPRINTF("EDT SIze %d\n", header->size);

        for(j=0; j<depc; j++)
        {
            header = (struct hiveHeader *) hiveRouteTableLookupItem( (hiveGuid_t) depv[j].guid );
            res = false;

            if(header != NULL && !res)
            {
                DPRINTF("Edt guid not found sent %ld\n", depv[j].guid);
                DPRINTF("Sent depc %d %d\n",j, header->size);
                depv[j].ptr = (void*)(u64)0x2;
                memcpy( ptr, header, header->size );
                ptr+=header->size;
                finalSize+=header->size;
                DPRINTF("Edt guid not found sent %ld %p\n", depv[j].guid, depv[j].ptr);
            }
            else if(header != NULL && res)
            {
                depv[j].ptr = (void*)(u64)0x1;
                DPRINTF("Edt guid found sent %ld %p\n", depv[j].guid, depv[j].ptr);

            }
            else
                DPRINTF("Edt guid error sent %ld %p\n", depv[j].guid, depv[j].ptr);
        }

    }
    packet->size = finalSize;
    DPRINTF("tog %d %d %d\n", finalSize, finalSize-sizeof(struct hiveRemotePacket), ((char *)ptr)-((char *)packet));

    DPRINTF("------------Packaging edts done-----------\n");
    *package = packet;

    return finalSize;
}

unsigned int handleIncomingEdts( char* address, int edtSizes )
{
    struct hiveHeader * header;
    int size = 0;
    struct hiveDb * db;
    int i, totalSize=0;
    void * newEdt;
    struct hiveEdt * edt;
    u32 depc;
    hiveEdtDep_t *depv;
    unsigned int totalEdtsRecieved = 0;

    while(totalSize != edtSizes)
    {
        header = (struct hiveHeader *) (address+totalSize);
        //if(header->size != 192 && header->size != 240)
        DPRINTF("%d\n", header->size);

        totalSize += header->size;
        HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
        newEdt = hiveMalloc( header->size );
        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
        memcpy(newEdt, header, header->size);
        
        edt = newEdt;
        hiveRouteTableAddItemRace( edt, (hiveGuid_t) edt->currentEdt, hiveGlobalRankId, false);
        depc = edt->depc;
        depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);

        DPRINTF("Edt Stolen needs %d\n", depc);

        for(i=0; i< depc; i++)
           depv[i].ptr=NULL;

        //if(((u64)edt->currentEdt) < 0x16)
        //    hiveDebugGenerateSegFault();
        
//        edt->ewSortList = 0x0;
        //    hiveDebugGenerateSegFault();
//        PRINTF("STOLE %p %lu %lu\n", edt->funcPtr, edt->currentEdt, edt->outputEvent);
        hiveHandleReadyEdt( newEdt );
        HIVECOUNTERINCREMENT(remoteEdtReceived);
        totalEdtsRecieved++;
    }
    return totalEdtsRecieved;
}
void handleIncomingEdtsAndDbs( char* address, int edtSizes )
{
    struct hiveHeader * header;
    int size = 0;
    struct hiveDb * db;
    int i, totalSize=0;
    void * newEdt;
    struct hiveEdt * edt;
    u32 depc;
    hiveEdtDep_t *depv;
    void * newDb;
    struct hiveDb * dbRes;
    DPRINTF("---------------Edts Incoming-----------------\n");
    while(totalSize != edtSizes)
    {
        header = (struct hiveHeader *) (address+totalSize);

        totalSize += header->size;
        HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
        newEdt = hiveMalloc( header->size );
        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
        memcpy(newEdt, header, header->size);

        edt = newEdt;
        depc = edt->depc;
        depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
        DPRINTF("Edt Stolen %d %d\n", header->size, edtSizes);
        if(depc == 0 )
        {
            DPRINTF("gerror\n");
        }

        for(i=0; i< depc; i++)
        {
            if( depv[i].ptr != NULL)
            {
                DPRINTF("sdsd %p\n", depv[i].ptr);
                void * tPtr = NULL;
                if( tPtr == NULL)
                {

                    if(depv[i].ptr == (void *) 0x2 )
                    {
                        DPRINTF("Edt guid not found %ld\n", depv[i].guid);
                        header = (struct hiveHeader *) (address+totalSize);
                        DPRINTF("Edt Stolen dep %d %d\n", i, header->size);
                        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
                        newDb = hiveMalloc(header->size);
                        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
                        memcpy( newDb, header, header->size );
                        dbRes =newDb;

                        depv[i].ptr = dbRes+1;

                        totalSize += header->size;

                        depv[i].guid = hiveGuidCreate(newDb);
                        DPRINTF("gll %ld %p\n", depv[i].guid, depv[i].ptr);
                    }
                    else if(depv[i].ptr == (void *) 0x1 )
                    {
                        DPRINTF("Edt Stolen fail %d\n", i);
                        depv[i].ptr=NULL;
                    }
                    else
                    {
                        DPRINTF("Edt Stolen fail2 %d %ld\n", i, depv[i].guid);
                        void * ptr = hiveRouteTableLookupItem( (hiveGuid_t) depv[i].guid );
                        depv[i].ptr=ptr;

                        if(ptr != NULL)
                            depv[i].ptr = ((struct hiveDb*)ptr)+1;
                    }
                }
                else
                {
                    DPRINTF("Rocal %p\n", depv[i].ptr);
                    if(depv[i].ptr != (void *) 0x1 )
                    {
                        DPRINTF("Houston\n");
                    }
                    else
                    {
                        DPRINTF("Error 12334\n");
                    }
                    depv[i].ptr = tPtr;

                }
                DPRINTF("depv[i].ptr = %p\n", depv[i].ptr);
            }
            DPRINTF("tot %d\n",totalSize);
        }
        DPRINTF("tota %d\n",totalSize);
        hiveHandleReadyEdt( newEdt );
    }
    DPRINTF("%d %d\n", totalSize, edtSizes);
    DPRINTF("---------------Edts Done-----------------\n");
}

void hiveRemoteMetricUpdate(int rank, int type, int level, u64 timeStamp, u64 toAdd, bool sub)
{
    DPRINTF("Remote Metric Update");
    struct hiveRemoteMetricUpdate packet; 
    packet.type = type;
    packet.timeStamp = timeStamp;
    packet.toAdd = toAdd;
    packet.sub = sub; 
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_REMOTE_METRIC_UPDATE);
    hiveRemoteSendRequestAsync( rank, (char *)&packet, sizeof(packet) );
}

void hiveRemoteHandleActiveMessage(void * ptr)
{
    struct hiveRemoteMemoryMovePacket * packet = ptr;    
    struct hiveEdt * edtOrig = (struct hiveEdt *)(packet+1);
    unsigned int edtSize = edtOrig->header.size;
    
    HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
    struct hiveEdt * edt = hiveMalloc(edtSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    memcpy(edt, edtOrig, edtSize);

    unsigned int size = packet->header.size - sizeof(struct hiveRemoteMemoryMovePacket) - edtSize;
    struct hiveDb * dbOrig = (struct hiveDb *)(((char*)(edtOrig)) + edtSize);
    hiveEdtDep_t * edtDep = (hiveEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
    while(size > 0)
    {
//        PRINTF("unpack size %u\n", size);
        struct hiveDb * db = hiveRouteTableLookupItem(edtDep->guid);
        if(db)
            memcpy(db, dbOrig, dbOrig->header.size);
        else
        {
            HIVESETMEMSHOTTYPE(hiveDbMemorySize);
            db = hiveMalloc(dbOrig->header.size);
            HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
            memcpy(db, dbOrig, dbOrig->header.size);
        }
//        hiveRouteTableUpdateItem(db, edtDep->guid, hiveGlobalRankId);
        edtDep->ptr = db;
        edt->depcNeeded--;
        
        size-=dbOrig->header.size;
        edtDep++;
        dbOrig = (struct hiveDb*)((char*)(dbOrig) + dbOrig->header.size);
    }

    if(edt->depcNeeded == 0)
    {
        hiveRouteTableAddItemRace(edt, (hiveGuid_t)packet->guid, hiveGlobalRankId, false);
        hiveHandleReadyEdt(edt);
    }
    else
    {    
        hiveRouteTableAddItemRace(edt, (hiveGuid_t) packet->guid, hiveGlobalRankId, false);
        hiveRouteTableFireOO(packet->guid, hiveOutOfOrderHandler); 
    }            
}

void hiveRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free)
{
    if(rank==hiveGlobalRankId)
    {
        funPtr(args);
        if(free)
            hiveFree(args);
        return;
    }
    struct hiveRemoteSend packet;
    packet.funPtr = funPtr;
    int totalSize = sizeof(struct hiveRemoteSend)+size;
    hiveFillPacketHeader(&packet.header, totalSize, HIVE_REMOTE_SEND);
    
    if(free)
        hiveRemoteSendRequestPayloadAsyncFree(rank, (char*)&packet, sizeof(packet), (char *)args, 0, size, NULL_GUID, hiveFree);
    else
        hiveRemoteSendRequestPayloadAsync(rank, (char *)&packet, sizeof(packet), (char *)args, size);
}

void hiveRemoteHandleSend(void * pack)
{
    struct hiveRemoteSend * packet = pack;
    void * args = (void*)(packet+1);
    packet->funPtr(args);
}

void hiveRemoteEpochInitSend(unsigned int rank, hiveGuid_t epochGuid, hiveGuid_t edtGuid, unsigned int slot)
{
    DPRINTF("Net Epoch Init Send: %u\n", rank);
    struct hiveRemoteEpochInitPacket packet;
    packet.epochGuid = epochGuid;
    packet.edtGuid = edtGuid;
    packet.slot = slot;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_EPOCH_INIT);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteHandleEpochInitSend(void * pack)
{
    DPRINTF("Net Epoch Init Rec\n");
    struct hiveRemoteEpochInitPacket * packet = pack;
    createEpoch(&packet->epochGuid, packet->edtGuid, packet->slot);
}

void hiveRemoteEpochReq(unsigned int rank, hiveGuid_t guid)
{
    DPRINTF("Net Epoch Req Send: %u\n", rank);
    struct hiveRemoteEpochReqPacket packet;
    packet.epochGuid = guid;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_EPOCH_REQ);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteHandleEpochReq(void * pack)
{
    DPRINTF("Net Epoch Req Rec\n");
    struct hiveRemoteEpochReqPacket * packet = pack;
    //For now the source and dest are the same...
    sendEpoch(packet->epochGuid, packet->header.rank, packet->header.rank);
}

void hiveRemoteEpochSend(unsigned int rank, hiveGuid_t guid, unsigned int active, unsigned int finish)
{
    DPRINTF("Net Epoch Send Send: %u\n", rank);
    struct hiveRemoteEpochSendPacket packet;
    packet.epochGuid = guid;
    packet.active = active;
    packet.finish = finish;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_EPOCH_SEND);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteHandleEpochSend(void * pack)
{
    DPRINTF("Net Epoch Send: Rec\n");
    struct hiveRemoteEpochSendPacket * packet = pack;
    reduceEpoch(packet->epochGuid, packet->active, packet->finish);
}

void hiveRemoteAtomicAddInArrayDb(unsigned int rank, hiveGuid_t dbGuid, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct hiveRemoteAtomicAddInArrayDbPacket packet;
    packet.dbGuid = dbGuid;
    packet.edtGuid = edtGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.index = index;
    packet.toAdd = toAdd;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_ATOMIC_ADD_ARRAYDB);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteHandleAtomicAddInArrayDb(void * pack)
{
    struct hiveRemoteAtomicAddInArrayDbPacket * packet = pack;
    struct hiveDb * db = hiveRouteTableLookupItem(packet->dbGuid);
    internalAtomicAddInArrayDb(packet->dbGuid, packet->index, packet->toAdd, packet->edtGuid, packet->slot, packet->epochGuid);
    
}

void hiveRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, hiveGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct hiveRemoteAtomicCompareAndSwapInArrayDbPacket packet;
    packet.dbGuid = dbGuid;
    packet.edtGuid = edtGuid;
    packet.epochGuid = epochGuid;
    packet.slot = slot;
    packet.index = index;
    packet.oldValue = oldValue;
    packet.newValue = newValue;
    hiveFillPacketHeader(&packet.header, sizeof(packet), HIVE_ATOMIC_CAS_ARRAYDB);
    hiveRemoteSendRequestAsync(rank, (char *)&packet, sizeof(packet));
}

void hiveRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack)
{
    struct hiveRemoteAtomicCompareAndSwapInArrayDbPacket * packet = pack;
    struct hiveDb * db = hiveRouteTableLookupItem(packet->dbGuid);
    internalAtomicCompareAndSwapInArrayDb(packet->dbGuid, packet->index, packet->oldValue, packet->newValue, packet->edtGuid, packet->slot, packet->epochGuid);
    
}