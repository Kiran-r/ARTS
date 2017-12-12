#include "hive.h"
#include "hiveUtil.h"
#include "hiveOutOfOrder.h"
#include "hiveOutOfOrderList.h"
#include "hiveRuntime.h"
#include "hiveMalloc.h"
#include "hiveGlobals.h"
#include "hiveRouteTable.h"
#include "hiveEdtFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveRemoteFunctions.h"
#include "hiveGuid.h"
#include "hiveAtomics.h"
#include <string.h>
#include <unistd.h>

#define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

enum hiveOutOfOrderType
{
    ooSignalEdt,
    ooEventSatisfy,
    ooEventSatisfySlot,
    ooAddDependence,
    ooHandleReadyEdt,
    ooRemoteDbSend,
    ooDbRequestSatisfy,
    ooDbFullSend,
    ooGetFromDb,
    ooSignalEdtPtr
};

struct ooSignalEdt
{
    enum hiveOutOfOrderType type;
    hiveGuid_t edtPacket;
    hiveGuid_t dataGuid;
    u32 slot;
    hiveDbAccessMode_t mode;
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
    hiveDbAccessMode_t mode;
};

struct ooEventSatisfy
{
    enum hiveOutOfOrderType type;
    hiveGuid_t eventGuid;
    hiveGuid_t dataGuid;
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
    hiveDbAccessMode_t mode;
    hiveGuid_t dataGuid;
};

struct ooRemoteDbFullSend
{
    enum hiveOutOfOrderType type;
    int rank;
    struct hiveEdt * edt;
    unsigned int slot;
    hiveDbAccessMode_t mode;
};

struct ooRemoteGetFromDb
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
            hiveSignalEdt(edt->edtPacket, edt->dataGuid, edt->slot, edt->mode);
            break;
        }
        case ooEventSatisfy:
        {
            struct ooEventSatisfy * event = handleMe;
            hiveEventSatisfy( event->eventGuid, event->dataGuid );
            break;
        }
        case ooEventSatisfySlot:
        {
            struct ooEventSatisfySlot * event = handleMe;
            hiveEventSatisfySlot( event->eventGuid, event->dataGuid, event->slot );
            break;
        }
        case ooAddDependence:
        {
            struct ooAddDependence * dep = handleMe;
            hiveAddDependence( dep->source, dep->destination, dep->slot, dep->mode );
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
            struct ooRemoteGetFromDb * req = handleMe;
            hiveGetFromDb(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size);
            break;
            
        }
        case ooSignalEdtPtr:
        {
            struct ooSignalEdtPtr * req = handleMe;
            hiveSignalEdtPtr(req->edtGuid, req->dbGuid, req->ptr, req->size, req->slot);
            break;
        }
        default:
            PRINTF("OO Handler Error\n");
    }
    hiveFree(handleMe);

}

void hiveOutOfOrderSignalEdt (hiveGuid_t waitOn, hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveDbAccessMode_t mode)
{
    HIVECOUNTERTIMERSTART(ooSigEdt);
    struct ooSignalEdt * edt = hiveMalloc(sizeof(struct ooSignalEdt));
    edt->type = ooSignalEdt;
    edt->edtPacket = edtPacket;
    edt->dataGuid = dataGuid;
    edt->slot = slot;
    edt->mode = mode;
    bool res =  hiveRouteTableAddOO(waitOn, edt, hiveGuidGetRank(edtPacket));
    if(!res)
    {
        hiveSignalEdt( edtPacket, dataGuid, slot, mode );
        hiveFree(edt);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooSigEdt);    
}

void hiveOutOfOrderEventSatisfy(hiveGuid_t waitOn, hiveGuid_t eventGuid, hiveGuid_t dataGuid )
{
    HIVECOUNTERTIMERSTART(ooEventSat);
    struct ooEventSatisfy * event = hiveMalloc( sizeof(struct ooEventSatisfy) );
    event->type = ooEventSatisfy;
    event->eventGuid = eventGuid;
    event->dataGuid = dataGuid;
    bool res =  hiveRouteTableAddOO( waitOn, event, hiveGuidGetRank(eventGuid) );
    if(!res)
    {
        hiveEventSatisfy( eventGuid, dataGuid );
        hiveFree(event);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooEventSat);
}

void hiveOutOfOrderEventSatisfySlot(hiveGuid_t waitOn, hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot )
{
    HIVECOUNTERTIMERSTART(ooEventSatSlot);
    struct ooEventSatisfySlot * event = hiveMalloc( sizeof(struct ooEventSatisfySlot) );
    event->type = ooEventSatisfySlot;
    event->eventGuid = eventGuid;
    event->dataGuid = dataGuid;
    event->slot = slot;
    bool res =  hiveRouteTableAddOO(waitOn, event, hiveGuidGetRank(eventGuid));
    if(!res)
    {
        hiveEventSatisfySlot(eventGuid, dataGuid, slot);
        hiveFree(event);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooEventSatSlot);
}

void hiveOutOfOrderAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode, hiveGuid_t waitOn)
{
    HIVECOUNTERTIMERSTART(ooAddDep);
    struct ooAddDependence * dep = hiveMalloc(sizeof(struct ooAddDependence));
    dep->type = ooAddDependence;
    dep->source = source;
    dep->destination = destination;
    dep->slot = slot;
    dep->mode = mode;
    bool res = hiveRouteTableAddOO(waitOn, dep, hiveGuidGetRank(waitOn));
    if(!res)
    {
        hiveAddDependence(source, destination, slot, mode);
        hiveFree(dep);
    }    
    HIVECOUNTERTIMERENDINCREMENT(ooAddDep);
}

void hiveOutOfOrderHandleReadyEdt(hiveGuid_t triggerGuid, struct hiveEdt *edt)
{
    HIVECOUNTERTIMERSTART(ooReadyEdt);
    struct ooHandleReadyEdt * readyEdt = hiveMalloc(sizeof(struct ooHandleReadyEdt));
    readyEdt->type = ooHandleReadyEdt;
    readyEdt->edt = edt;
    bool res = hiveRouteTableAddOO(triggerGuid, readyEdt, hiveGuidGetRank(triggerGuid));
    if(!res)
    {
        hiveHandleReadyEdt(edt);
        hiveFree(readyEdt);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooReadyEdt);
}

void hiveOutOfOrderHandleRemoteDbSend(int rank, hiveGuid_t dbGuid, hiveDbAccessMode_t mode)
{
    HIVECOUNTERTIMERSTART(ooRemoteDb);
    struct ooRemoteDbSend * readySend = hiveMalloc(sizeof(struct ooRemoteDbSend));
    readySend->type = ooRemoteDbSend;
    readySend->rank = rank;
    readySend->dataGuid = dbGuid;
    readySend->mode = mode;
    bool res = hiveRouteTableAddOO(dbGuid, readySend, hiveGuidGetRank(dbGuid));
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveRemoteDbSendCheck(readySend->rank, db, readySend->mode);
        hiveFree(readySend);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooRemoteDb);
}

void hiveOutOfOrderHandleDbRequest(hiveGuid_t dbGuid, struct hiveEdt *edt, unsigned int slot)
{
    HIVECOUNTERTIMERSTART(ooRemoteDb);
    struct ooDbRequestSatisfy * req = hiveMalloc(sizeof(struct ooDbRequestSatisfy));
    req->type = ooDbRequestSatisfy;
    req->edt = edt;
    req->slot = slot;
    bool res = hiveRouteTableAddOO(dbGuid, req, hiveGuidGetRank(dbGuid));
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveDbRequestCallback(req->edt, req->slot, db);
        hiveFree(req);
    }
    HIVECOUNTERTIMERENDINCREMENT(ooRemoteDb);
}

//This should save one lookup compared to the function above...
void hiveOutOfOrderHandleDbRequestWithOOList(struct hiveOutOfOrderList * addToMe, void ** data, struct hiveEdt *edt, unsigned int slot)
{
//    HIVECOUNTERTIMERSTART(ooRemoteDb);
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
//    HIVECOUNTERTIMERENDINCREMENT(ooRemoteDb);
}

void hiveOutOfOrderHandleRemoteDbFullSend(hiveGuid_t dbGuid, int rank, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
//    HIVECOUNTERTIMERSTART(ooReadyEdt);
    struct ooRemoteDbFullSend * dbSend = hiveMalloc(sizeof(struct ooRemoteDbFullSend));
    dbSend->type = ooDbFullSend;
    dbSend->rank = rank;
    dbSend->edt = edt;
    dbSend->slot = slot;
    dbSend->mode = mode;
    bool res = hiveRouteTableAddOO(dbGuid, dbSend, hiveGuidGetRank(dbGuid));
    if(!res)
    {
        struct hiveDb * db = hiveRouteTableLookupItem(dbGuid);
        hiveRemoteDbFullSendCheck(dbSend->rank, db, dbSend->edt, dbSend->slot, dbSend->mode);
        hiveFree(dbSend);
    }
//    HIVECOUNTERTIMERENDINCREMENT(ooReadyEdt);
}

void hiveOutOfOrderGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    struct ooRemoteGetFromDb * req = hiveMalloc(sizeof(struct ooRemoteGetFromDb));
    req->type = ooGetFromDb;
    req->edtGuid = edtGuid;
    req->dbGuid = dbGuid;
    req->slot = slot;
    req->offset = offset;
    req->size = size;
    bool res = hiveRouteTableAddOO(dbGuid, req, hiveGuidGetRank(dbGuid));
    if(!res)
    {
        hiveGetFromDb(req->edtGuid, req->dbGuid, req->slot, req->offset, req->size);
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
    bool res = hiveRouteTableAddOO(edtGuid, req, hiveGuidGetRank(dbGuid));
    if(!res)
    {
        hiveSignalEdtPtr(req->edtGuid, req->dbGuid, req->ptr, req->size, req->slot);
        hiveFree(req);
    }
}