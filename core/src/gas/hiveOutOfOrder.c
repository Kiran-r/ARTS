#include "hive.h"
#include "hiveUtil.h"
#include "hiveOutOfOrder.h"
#include "hiveOutOfOrderList.h"
#include "hiveRuntime.h"
#include "hiveMalloc.h"
#include "hiveGlobals.h"
#include "hiveRouteTable.h"
#include "hiveEdtFunctions.h"
#include "hiveEventFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveRemoteFunctions.h"
#include "hiveGuid.h"
#include "hiveAtomics.h"
#include "hiveTerminationDetection.h"
#include "hiveArrayDb.h"
#include <string.h>
#include <unistd.h>

#define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

enum hiveOutOfOrderType
{
    ooSignalEdt,
    ooEventSatisfySlot,
    ooAddDependence,
    ooHandleReadyEdt,
    ooRemoteDbSend,
    ooDbRequestSatisfy,
    ooDbFullSend,
    ooGetFromDb,
    ooSignalEdtPtr,
    ooPutInDb,
    ooEpochActive,
    ooEpochFinish,
    ooEpochSend,
    ooEpochIncQueue,
    ooAtomicAddInArrayDb,
    ooAtomicCompareAndSwapInArrayDb,
    ooDbMove
};

struct ooSignalEdt
{
    enum hiveOutOfOrderType type;
    hiveGuid_t edtPacket;
    hiveGuid_t dataGuid;
    u32 slot;
    hiveType_t mode;
};

struct ooDbRequestSatisfy
{
    enum hiveOutOfOrderType type;
    struct hiveEdt * edt;
    u32 slot;
};

struct ooAddDependence
{
    enum hiveOutOfOrderType type;
    hiveGuid_t source;
    hiveGuid_t destination;
    u32 slot;
    hiveType_t mode;
};

struct ooEventSatisfySlot
{
    enum hiveOutOfOrderType type;
    hiveGuid_t eventGuid;
    hiveGuid_t dataGuid;
    u32 slot;
};

struct ooHandleReadyEdt
{
    enum hiveOutOfOrderType type;
    struct hiveEdt *edt;
};

struct ooRemoteDbSend
{
    enum hiveOutOfOrderType type;
    int rank;
    hiveType_t mode;
    hiveGuid_t dataGuid;
};

struct ooRemoteDbFullSend
{
    enum hiveOutOfOrderType type;
    int rank;
    struct hiveEdt * edt;
    unsigned int slot;
    hiveType_t mode;
};

struct ooGetFromDb
{
    enum hiveOutOfOrderType type;
    hiveGuid_t edtGuid;
    hiveGuid_t dbGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct ooSignalEdtPtr
{
    enum hiveOutOfOrderType type;
    hiveGuid_t edtGuid;
    hiveGuid_t dbGuid;
    void * ptr;
    unsigned int size;
    unsigned int slot;
};

struct ooPutInDb
{
    enum hiveOutOfOrderType type;
    void * ptr;
    hiveGuid_t edtGuid;
    hiveGuid_t dbGuid;
    hiveGuid_t epochGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct ooEpoch
{
    enum hiveOutOfOrderType type;
    hiveGuid_t guid;
};

struct ooEpochSend
{
    enum hiveOutOfOrderType type;
    hiveGuid_t guid;
    unsigned int source;
    unsigned int dest;
};

struct ooAtomicAddInArrayDb
{
    enum hiveOutOfOrderType type;
    hiveGuid_t dbGuid;
    hiveGuid_t edtGuid;
    hiveGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int toAdd;
};

struct ooAtomicCompareAndSwapInArrayDb
{
    enum hiveOutOfOrderType type;
    hiveGuid_t dbGuid;
    hiveGuid_t edtGuid;
    hiveGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int oldValue;
    unsigned int newValue;
};

struct ooGeneric
{
    enum hiveOutOfOrderType type;
};

inline void hiveOutOfOrderHandler(void * handleMe, void * memoryPtr)
{
    struct ooGeneric * typePtr = handleMe;
    switch(typePtr->type)
    {
        case ooSignalEdt:
        {
            struct ooSignalEdt * edt = handleMe;
            internalSignalEdt(edt->edtPacket, edt->slot, edt->dataGuid, edt->mode, NULL, 0);
            break;
        }
        case ooEventSatisfySlot:
        {
            struct ooEventSatisfySlot * event = handleMe;
            hiveEventSatisfySlot(event->eventGuid, event->dataGuid, event->slot);
            break;
        }
        case ooAddDependence:
        {
            struct ooAddDependence * dep = handleMe;
            hiveAddDependence(dep->source, dep->destination, dep->slot);
            break;
        }
        case ooHandleReadyEdt:
        {
            struct ooHandleReadyEdt * readyEdt = handleMe;
            hiveHandleReadyEdt( readyEdt->edt );
            break;
        }
        case ooRemoteDbSend:
        {
            struct ooRemoteDbSend * dbSend = handleMe;
            hiveRemoteDbSendCheck(dbSend->rank, (struct hiveDb *)memoryPtr, dbSend->mode);
            break;
        }
        case ooDbRequestSatisfy:
        {
            struct ooDbRequestSatisfy * req = handleMe;
            hiveDbRequestCallback(req->edt, req->slot, (struct hiveDb *)memoryPtr);
            break;
        }
        case ooDbFullSend:
        {
            struct ooRemoteDbFullSend * dbSend = handleMe;
            hiveRemoteDbFullSendCheck(dbSend->rank, (struct hiveDb *)memoryPtr, dbSend->edt, dbSend->slot, dbSend->mode);
            break;
        }
        case ooGetFromDb:
        {
            struct ooGetFromDb * req = handleMe;
            hiveGetFromDbAt(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, hiveGlobalRankId);
            break;
            
        }
        case ooSignalEdtPtr:
        {
            struct ooSignalEdtPtr * req = handleMe;
            hiveSignalEdtPtr(req->edtGuid, req->slot, req->ptr, req->size);
            break;
        }
        case ooPutInDb:
        {
            struct ooPutInDb * req = handleMe;
            internalPutInDb(req->ptr, req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, req->epochGuid, hiveGlobalRankId);
            hiveFree(req->ptr);
            break;
            
        }
        case ooEpochActive:
        {
//            PRINTF("ooActveFire\n");
            struct ooEpoch * req = handleMe;
            incrementActiveEpoch(req->guid);
            break;
        }
        case ooEpochFinish:
        {
//            PRINTF("ooFinishFire\n");
            struct ooEpoch * req = handleMe;
            incrementFinishedEpoch(req->guid);
            break;
        }
        case ooEpochSend:
        {
//            PRINTF("ooEpochSendFire\n");
            struct ooEpochSend * req = handleMe;
            sendEpoch(req->guid, req->source, req->dest);
            break;
        }
        case ooEpochIncQueue:
        {
            struct ooEpoch * req = handleMe;
            incrementQueueEpoch(req->guid);
            break;
        }
        case ooAtomicAddInArrayDb:
        {
            struct ooAtomicAddInArrayDb * req = handleMe;
            internalAtomicAddInArrayDb(req->dbGuid, req->index, req->toAdd, req->edtGuid, req->slot, req->epochGuid);
            break;
        }
        case ooAtomicCompareAndSwapInArrayDb:
        {
            struct ooAtomicCompareAndSwapInArrayDb * req = handleMe;
            internalAtomicCompareAndSwapInArrayDb(req->dbGuid, req->index, req->oldValue, req->newValue, req->edtGuid, req->slot, req->epochGuid);
        }
        case ooDbMove:
        {
            struct ooRemoteDbSend * req = handleMe;
            hiveDbMove(req->dataGuid, req->rank);
            break;
        }
        default:
            PRINTF("OO Handler Error\n");
    }
    hiveFree(handleMe);

}

void hiveOutOfOrderSignalEdt (hiveGuid_t waitOn, hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveType_t mode)
{
    struct ooSignalEdt * edt = hiveMalloc(sizeof(struct ooSignalEdt));
    edt->type = ooSignalEdt;
    edt->edtPacket = edtPacket;
    edt->dataGuid = dataGuid;
    edt->slot = slot;
    edt->mode = mode;
    bool res =  hiveRouteTableAddOO(waitOn, edt);
    if(!res)
    {
        internalSignalEdt(edtPacket, slot, dataGuid, mode, NULL, 0);
        hiveFree(edt);
    }   
}

void hiveOutOfOrderEventSatisfySlot(hiveGuid_t waitOn, hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot )
{
    struct ooEventSatisfySlot * event = hiveMalloc( sizeof(struct ooEventSatisfySlot) );
    event->type = ooEventSatisfySlot;
    event->eventGuid = eventGuid;
    event->dataGuid = dataGuid;
    event->slot = slot;
    bool res =  hiveRouteTableAddOO(waitOn, event);
    if(!res)
    {
        hiveEventSatisfySlot(eventGuid, dataGuid, slot);
        hiveFree(event);
    }
}

void hiveOutOfOrderAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveType_t mode, hiveGuid_t waitOn)
{
    struct ooAddDependence * dep = hiveMalloc(sizeof(struct ooAddDependence));
    dep->type = ooAddDependence;
    dep->source = source;
    dep->destination = destination;
    dep->slot = slot;
    dep->mode = mode;
    bool res = hiveRouteTableAddOO(waitOn, dep);
    if(!res)
    {
        hiveAddDependence(source, destination, slot);
        hiveFree(dep);
    }
}

void hiveOutOfOrderHandleReadyEdt(hiveGuid_t triggerGuid, struct hiveEdt *edt)
{
    struct ooHandleReadyEdt * readyEdt = hiveMalloc(sizeof(struct ooHandleReadyEdt));
    readyEdt->type = ooHandleReadyEdt;
    readyEdt->edt = edt;
    bool res = hiveRouteTableAddOO(triggerGuid, readyEdt);
    if(!res)
    {
        hiveHandleReadyEdt(edt);
        hiveFree(readyEdt);
    }
}

void hiveOutOfOrderHandleRemoteDbSend(int rank, hiveGuid_t dbGuid, hiveType_t mode)
{
    struct ooRemoteDbSend * readySend = hiveMalloc(sizeof(struct ooRemoteDbSend));
    readySend->type = ooRemoteDbSend;
    readySend->rank = rank;
    readySend->dataGuid = dbGuid;
    readySend->mode = mode;
    bool res = hiveRouteTableAddOO(dbGuid, readySend);
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveRemoteDbSendCheck(readySend->rank, db, readySend->mode);
        hiveFree(readySend);
    }
}

void hiveOutOfOrderHandleDbRequest(hiveGuid_t dbGuid, struct hiveEdt *edt, unsigned int slot)
{
    struct ooDbRequestSatisfy * req = hiveMalloc(sizeof(struct ooDbRequestSatisfy));
    req->type = ooDbRequestSatisfy;
    req->edt = edt;
    req->slot = slot;
    bool res = hiveRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveDbRequestCallback(req->edt, req->slot, db);
        hiveFree(req);
    }
}

//This should save one lookup compared to the function above...
void hiveOutOfOrderHandleDbRequestWithOOList(struct hiveOutOfOrderList * addToMe, void ** data, struct hiveEdt *edt, unsigned int slot)
{
    struct ooDbRequestSatisfy * req = hiveMalloc(sizeof(struct ooDbRequestSatisfy));
    req->type = ooDbRequestSatisfy;
    req->edt = edt;
    req->slot = slot;
    bool res = hiveOutOfOrderListAddItem(addToMe, req);
    if(!res)
    {
        hiveDbRequestCallback(req->edt, req->slot, *data);
        hiveFree(req);
    }
}

void hiveOutOfOrderHandleRemoteDbFullSend(hiveGuid_t dbGuid, int rank, struct hiveEdt * edt, unsigned int slot, hiveType_t mode)
{
    struct ooRemoteDbFullSend * dbSend = hiveMalloc(sizeof(struct ooRemoteDbFullSend));
    dbSend->type = ooDbFullSend;
    dbSend->rank = rank;
    dbSend->edt = edt;
    dbSend->slot = slot;
    dbSend->mode = mode;
    bool res = hiveRouteTableAddOO(dbGuid, dbSend);
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveRemoteDbFullSendCheck(dbSend->rank, db, dbSend->edt, dbSend->slot, dbSend->mode);
        hiveFree(dbSend);
    }
}

void hiveOutOfOrderGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    struct ooGetFromDb * req = hiveMalloc(sizeof(struct ooGetFromDb));
    req->type = ooGetFromDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->slot = slot;
    req->offset = offset;
    req->size = size;
    bool res = hiveRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        hiveGetFromDbAt(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, hiveGlobalRankId);
        hiveFree(req);
    }
}

void hiveOutOfOrderSignalEdtWithPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot)
{
    struct ooSignalEdtPtr * req = hiveMalloc(sizeof(struct ooSignalEdtPtr));
    req->type = ooSignalEdtPtr;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->size = size;
    req->slot = slot;
    req->ptr = ptr;
    bool res = hiveRouteTableAddOO(edtGuid, req);
    if(!res)
    {
        hiveSignalEdtPtr(req->edtGuid, req->slot, req->ptr, req->size);
        hiveFree(req);
    }
}

void hiveOutOfOrderPutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epochGuid)
{
    struct ooPutInDb * req = hiveMalloc(sizeof(struct ooPutInDb));
    req->type = ooPutInDb;
    req->ptr = ptr;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->slot = slot;
    req->offset = offset;
    req->size = size;
    req->epochGuid = epochGuid;
    bool res = hiveRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        internalPutInDb(req->ptr, req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, req->epochGuid, hiveGlobalRankId);
        hiveFree(req->ptr);
        hiveFree(req);
    }
}

void hiveOutOfOrderIncActiveEpoch(hiveGuid_t epochGuid)
{
    struct ooEpoch * req = hiveMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochActive;
    req->guid = epochGuid;
    bool res =  hiveRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementActiveEpoch(epochGuid);
        hiveFree(req);
    }   
}

void hiveOutOfOrderIncFinishedEpoch(hiveGuid_t epochGuid)
{
    struct ooEpoch * req = hiveMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochFinish;
    req->guid = epochGuid;
    bool res =  hiveRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementFinishedEpoch(epochGuid);
        hiveFree(req);
    }   
}

void hiveOutOfOrderSendEpoch(hiveGuid_t epochGuid, unsigned int source, unsigned int dest)
{
    struct ooEpochSend * req = hiveMalloc(sizeof(struct ooEpochSend));
    req->type = ooEpochSend;
    req->source = source;
    req->dest = dest;
    bool res = hiveRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        sendEpoch(epochGuid, source, dest);
        hiveFree(req);
    }
}

void hiveOutOfOrderIncQueueEpoch(hiveGuid_t epochGuid)
{
    struct ooEpoch * req = hiveMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochIncQueue;
    req->guid = epochGuid;
    bool res =  hiveRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementQueueEpoch(epochGuid);
        hiveFree(req);
    }   
}

void hiveOutOfOrderAtomicAddInArrayDb(hiveGuid_t dbGuid,  unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct ooAtomicAddInArrayDb * req = hiveMalloc(sizeof(struct ooAtomicAddInArrayDb));
    req->type = ooAtomicAddInArrayDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->epochGuid = epochGuid;
    req->slot = slot;
    req->index = index;
    req->toAdd = toAdd;
    bool res = hiveRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        PRINTF("edtGuid OO2: %lu\n", req->edtGuid);
        internalAtomicAddInArrayDb(req->dbGuid, req->index, req->toAdd, req->edtGuid, req->slot, req->epochGuid);
        hiveFree(req);
    }
}

void hiveOutOfOrderAtomicCompareAndSwapInArrayDb(hiveGuid_t dbGuid,  unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid)
{
    struct ooAtomicCompareAndSwapInArrayDb * req = hiveMalloc(sizeof(struct ooAtomicCompareAndSwapInArrayDb));
    req->type = ooAtomicAddInArrayDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->epochGuid = epochGuid;
    req->slot = slot;
    req->index = index;
    req->oldValue = oldValue;
    req->newValue = newValue;
    bool res = hiveRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        PRINTF("edtGuid OO2: %lu\n", req->edtGuid);
        internalAtomicCompareAndSwapInArrayDb(req->dbGuid, req->index, req->oldValue, req->newValue, req->edtGuid, req->slot, req->epochGuid);
        hiveFree(req);
    }
}

void hiveOutOfOrderDbMove(hiveGuid_t dataGuid, unsigned int rank)
{
    struct ooRemoteDbSend * req = hiveMalloc(sizeof(struct ooRemoteDbSend));
    req->type = ooDbMove;
    req->dataGuid = dataGuid;
    req->rank = rank;
    bool res =  hiveRouteTableAddOO(dataGuid, req);
    if(!res)
    {
        hiveDbMove(dataGuid, rank);
        hiveFree(req);
    }
}