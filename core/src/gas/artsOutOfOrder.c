#include "arts.h"
#include "artsUtil.h"
#include "artsOutOfOrder.h"
#include "artsOutOfOrderList.h"
#include "artsRuntime.h"
#include "artsMalloc.h"
#include "artsGlobals.h"
#include "artsRouteTable.h"
#include "artsEdtFunctions.h"
#include "artsEventFunctions.h"
#include "artsDbFunctions.h"
#include "artsRemoteFunctions.h"
#include "artsGuid.h"
#include "artsAtomics.h"
#include "artsTerminationDetection.h"
#include "artsArrayDb.h"
#include <string.h>
#include <unistd.h>

#define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

enum artsOutOfOrderType
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
    enum artsOutOfOrderType type;
    artsGuid_t edtPacket;
    artsGuid_t dataGuid;
    uint32_t slot;
    artsType_t mode;
};

struct ooDbRequestSatisfy
{
    enum artsOutOfOrderType type;
    struct artsEdt * edt;
    uint32_t slot;
};

struct ooAddDependence
{
    enum artsOutOfOrderType type;
    artsGuid_t source;
    artsGuid_t destination;
    uint32_t slot;
    artsType_t mode;
};

struct ooEventSatisfySlot
{
    enum artsOutOfOrderType type;
    artsGuid_t eventGuid;
    artsGuid_t dataGuid;
    uint32_t slot;
};

struct ooHandleReadyEdt
{
    enum artsOutOfOrderType type;
    struct artsEdt *edt;
};

struct ooRemoteDbSend
{
    enum artsOutOfOrderType type;
    int rank;
    artsType_t mode;
    artsGuid_t dataGuid;
};

struct ooRemoteDbFullSend
{
    enum artsOutOfOrderType type;
    int rank;
    struct artsEdt * edt;
    unsigned int slot;
    artsType_t mode;
};

struct ooGetFromDb
{
    enum artsOutOfOrderType type;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct ooSignalEdtPtr
{
    enum artsOutOfOrderType type;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    void * ptr;
    unsigned int size;
    unsigned int slot;
};

struct ooPutInDb
{
    enum artsOutOfOrderType type;
    void * ptr;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct ooEpoch
{
    enum artsOutOfOrderType type;
    artsGuid_t guid;
};

struct ooEpochSend
{
    enum artsOutOfOrderType type;
    artsGuid_t guid;
    unsigned int source;
    unsigned int dest;
};

struct ooAtomicAddInArrayDb
{
    enum artsOutOfOrderType type;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int toAdd;
};

struct ooAtomicCompareAndSwapInArrayDb
{
    enum artsOutOfOrderType type;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int oldValue;
    unsigned int newValue;
};

struct ooGeneric
{
    enum artsOutOfOrderType type;
};

inline void artsOutOfOrderHandler(void * handleMe, void * memoryPtr)
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
            artsEventSatisfySlot(event->eventGuid, event->dataGuid, event->slot);
            break;
        }
        case ooAddDependence:
        {
            struct ooAddDependence * dep = handleMe;
            artsAddDependence(dep->source, dep->destination, dep->slot);
            break;
        }
        case ooHandleReadyEdt:
        {
            struct ooHandleReadyEdt * readyEdt = handleMe;
            artsHandleReadyEdt( readyEdt->edt );
            break;
        }
        case ooRemoteDbSend:
        {
            struct ooRemoteDbSend * dbSend = handleMe;
            artsRemoteDbSendCheck(dbSend->rank, (struct artsDb *)memoryPtr, dbSend->mode);
            break;
        }
        case ooDbRequestSatisfy:
        {
            struct ooDbRequestSatisfy * req = handleMe;
            artsDbRequestCallback(req->edt, req->slot, (struct artsDb *)memoryPtr);
            break;
        }
        case ooDbFullSend:
        {
            struct ooRemoteDbFullSend * dbSend = handleMe;
            artsRemoteDbFullSendCheck(dbSend->rank, (struct artsDb *)memoryPtr, dbSend->edt, dbSend->slot, dbSend->mode);
            break;
        }
        case ooGetFromDb:
        {
            struct ooGetFromDb * req = handleMe;
            artsGetFromDbAt(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, artsGlobalRankId);
            break;
            
        }
        case ooSignalEdtPtr:
        {
            struct ooSignalEdtPtr * req = handleMe;
            artsSignalEdtPtr(req->edtGuid, req->slot, req->ptr, req->size);
            break;
        }
        case ooPutInDb:
        {
            struct ooPutInDb * req = handleMe;
            internalPutInDb(req->ptr, req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, req->epochGuid, artsGlobalRankId);
            artsFree(req->ptr);
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
            artsDbMove(req->dataGuid, req->rank);
            break;
        }
        default:
            PRINTF("OO Handler Error\n");
    }
    artsFree(handleMe);

}

void artsOutOfOrderSignalEdt (artsGuid_t waitOn, artsGuid_t edtPacket, artsGuid_t dataGuid, uint32_t slot, artsType_t mode)
{
    struct ooSignalEdt * edt = artsMalloc(sizeof(struct ooSignalEdt));
    edt->type = ooSignalEdt;
    edt->edtPacket = edtPacket;
    edt->dataGuid = dataGuid;
    edt->slot = slot;
    edt->mode = mode;
    bool res =  artsRouteTableAddOO(waitOn, edt);
    if(!res)
    {
        internalSignalEdt(edtPacket, slot, dataGuid, mode, NULL, 0);
        artsFree(edt);
    }   
}

void artsOutOfOrderEventSatisfySlot(artsGuid_t waitOn, artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot )
{
    struct ooEventSatisfySlot * event = artsMalloc( sizeof(struct ooEventSatisfySlot) );
    event->type = ooEventSatisfySlot;
    event->eventGuid = eventGuid;
    event->dataGuid = dataGuid;
    event->slot = slot;
    bool res =  artsRouteTableAddOO(waitOn, event);
    if(!res)
    {
        artsEventSatisfySlot(eventGuid, dataGuid, slot);
        artsFree(event);
    }
}

void artsOutOfOrderAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, artsGuid_t waitOn)
{
    struct ooAddDependence * dep = artsMalloc(sizeof(struct ooAddDependence));
    dep->type = ooAddDependence;
    dep->source = source;
    dep->destination = destination;
    dep->slot = slot;
    dep->mode = mode;
    bool res = artsRouteTableAddOO(waitOn, dep);
    if(!res)
    {
        artsAddDependence(source, destination, slot);
        artsFree(dep);
    }
}

void artsOutOfOrderHandleReadyEdt(artsGuid_t triggerGuid, struct artsEdt *edt)
{
    struct ooHandleReadyEdt * readyEdt = artsMalloc(sizeof(struct ooHandleReadyEdt));
    readyEdt->type = ooHandleReadyEdt;
    readyEdt->edt = edt;
    bool res = artsRouteTableAddOO(triggerGuid, readyEdt);
    if(!res)
    {
        artsHandleReadyEdt(edt);
        artsFree(readyEdt);
    }
}

void artsOutOfOrderHandleRemoteDbSend(int rank, artsGuid_t dbGuid, artsType_t mode)
{
    struct ooRemoteDbSend * readySend = artsMalloc(sizeof(struct ooRemoteDbSend));
    readySend->type = ooRemoteDbSend;
    readySend->rank = rank;
    readySend->dataGuid = dbGuid;
    readySend->mode = mode;
    bool res = artsRouteTableAddOO(dbGuid, readySend);
    if(!res)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        artsRemoteDbSendCheck(readySend->rank, db, readySend->mode);
        artsFree(readySend);
    }
}

void artsOutOfOrderHandleDbRequest(artsGuid_t dbGuid, struct artsEdt *edt, unsigned int slot)
{
    struct ooDbRequestSatisfy * req = artsMalloc(sizeof(struct ooDbRequestSatisfy));
    req->type = ooDbRequestSatisfy;
    req->edt = edt;
    req->slot = slot;
    bool res = artsRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        artsDbRequestCallback(req->edt, req->slot, db);
        artsFree(req);
    }
}

//This should save one lookup compared to the function above...
void artsOutOfOrderHandleDbRequestWithOOList(struct artsOutOfOrderList * addToMe, void ** data, struct artsEdt *edt, unsigned int slot)
{
    struct ooDbRequestSatisfy * req = artsMalloc(sizeof(struct ooDbRequestSatisfy));
    req->type = ooDbRequestSatisfy;
    req->edt = edt;
    req->slot = slot;
    bool res = artsOutOfOrderListAddItem(addToMe, req);
    if(!res)
    {
        artsDbRequestCallback(req->edt, req->slot, *data);
        artsFree(req);
    }
}

void artsOutOfOrderHandleRemoteDbFullSend(artsGuid_t dbGuid, int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode)
{
    struct ooRemoteDbFullSend * dbSend = artsMalloc(sizeof(struct ooRemoteDbFullSend));
    dbSend->type = ooDbFullSend;
    dbSend->rank = rank;
    dbSend->edt = edt;
    dbSend->slot = slot;
    dbSend->mode = mode;
    bool res = artsRouteTableAddOO(dbGuid, dbSend);
    if(!res)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        artsRemoteDbFullSendCheck(dbSend->rank, db, dbSend->edt, dbSend->slot, dbSend->mode);
        artsFree(dbSend);
    }
}

void artsOutOfOrderGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    struct ooGetFromDb * req = artsMalloc(sizeof(struct ooGetFromDb));
    req->type = ooGetFromDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->slot = slot;
    req->offset = offset;
    req->size = size;
    bool res = artsRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        artsGetFromDbAt(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, artsGlobalRankId);
        artsFree(req);
    }
}

void artsOutOfOrderSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot)
{
    struct ooSignalEdtPtr * req = artsMalloc(sizeof(struct ooSignalEdtPtr));
    req->type = ooSignalEdtPtr;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->size = size;
    req->slot = slot;
    req->ptr = ptr;
    bool res = artsRouteTableAddOO(edtGuid, req);
    if(!res)
    {
        artsSignalEdtPtr(req->edtGuid, req->slot, req->ptr, req->size);
        artsFree(req);
    }
}

void artsOutOfOrderPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid)
{
    struct ooPutInDb * req = artsMalloc(sizeof(struct ooPutInDb));
    req->type = ooPutInDb;
    req->ptr = ptr;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->slot = slot;
    req->offset = offset;
    req->size = size;
    req->epochGuid = epochGuid;
    bool res = artsRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        internalPutInDb(req->ptr, req->edtGuid, req->dbGuid, req->slot, req->offset, req->size, req->epochGuid, artsGlobalRankId);
        artsFree(req->ptr);
        artsFree(req);
    }
}

void artsOutOfOrderIncActiveEpoch(artsGuid_t epochGuid)
{
    struct ooEpoch * req = artsMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochActive;
    req->guid = epochGuid;
    bool res =  artsRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementActiveEpoch(epochGuid);
        artsFree(req);
    }   
}

void artsOutOfOrderIncFinishedEpoch(artsGuid_t epochGuid)
{
    struct ooEpoch * req = artsMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochFinish;
    req->guid = epochGuid;
    bool res =  artsRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementFinishedEpoch(epochGuid);
        artsFree(req);
    }   
}

void artsOutOfOrderSendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest)
{
    struct ooEpochSend * req = artsMalloc(sizeof(struct ooEpochSend));
    req->type = ooEpochSend;
    req->source = source;
    req->dest = dest;
    bool res = artsRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        sendEpoch(epochGuid, source, dest);
        artsFree(req);
    }
}

void artsOutOfOrderIncQueueEpoch(artsGuid_t epochGuid)
{
    struct ooEpoch * req = artsMalloc(sizeof(struct ooEpoch));
    req->type = ooEpochIncQueue;
    req->guid = epochGuid;
    bool res =  artsRouteTableAddOO(epochGuid, req);
    if(!res)
    {
        incrementQueueEpoch(epochGuid);
        artsFree(req);
    }   
}

void artsOutOfOrderAtomicAddInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct ooAtomicAddInArrayDb * req = artsMalloc(sizeof(struct ooAtomicAddInArrayDb));
    req->type = ooAtomicAddInArrayDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->epochGuid = epochGuid;
    req->slot = slot;
    req->index = index;
    req->toAdd = toAdd;
    bool res = artsRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        PRINTF("edtGuid OO2: %lu\n", req->edtGuid);
        internalAtomicAddInArrayDb(req->dbGuid, req->index, req->toAdd, req->edtGuid, req->slot, req->epochGuid);
        artsFree(req);
    }
}

void artsOutOfOrderAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid)
{
    struct ooAtomicCompareAndSwapInArrayDb * req = artsMalloc(sizeof(struct ooAtomicCompareAndSwapInArrayDb));
    req->type = ooAtomicAddInArrayDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->epochGuid = epochGuid;
    req->slot = slot;
    req->index = index;
    req->oldValue = oldValue;
    req->newValue = newValue;
    bool res = artsRouteTableAddOO(dbGuid, req);
    if(!res)
    {
        PRINTF("edtGuid OO2: %lu\n", req->edtGuid);
        internalAtomicCompareAndSwapInArrayDb(req->dbGuid, req->index, req->oldValue, req->newValue, req->edtGuid, req->slot, req->epochGuid);
        artsFree(req);
    }
}

void artsOutOfOrderDbMove(artsGuid_t dataGuid, unsigned int rank)
{
    struct ooRemoteDbSend * req = artsMalloc(sizeof(struct ooRemoteDbSend));
    req->type = ooDbMove;
    req->dataGuid = dataGuid;
    req->rank = rank;
    bool res =  artsRouteTableAddOO(dataGuid, req);
    if(!res)
    {
        artsDbMove(dataGuid, rank);
        artsFree(req);
    }
}