#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveGlobals.h"
#include "hiveAtomics.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveRuntime.h"
#include "hiveEdtFunctions.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveDebug.h"
#include <stdarg.h>
#include <string.h>

#define DPRINTF( ... )

__thread struct hiveEdt * currentEdt = NULL;

void hiveSetThreadLocalEdtInfo(struct hiveEdt * edt)
{
    hiveThreadInfo.currentEdtGuid = edt->currentEdt;
    currentEdt = edt;
}

bool hiveEdtCreateInternal(hiveGuid_t * guid, unsigned int route, unsigned int edtSpace, hiveGuid_t eventGuid, hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc)
{
    struct hiveEdt *edt;
    HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
    edt = (struct hiveEdt*)hiveCalloc(edtSpace);
    edt->header.type = HIVE_EDT;
    edt->header.size = edtSpace;
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
    if(edt)
    {
        bool createdGuid = false;
        if(*guid == NULL_GUID)
        {
            createdGuid = true;
            *guid = hiveGuidCreateForRank(route, HIVE_EDT);
        }

        edt->funcPtr = funcPtr;
        edt->depc = depc;
        edt->paramc = paramc;
        edt->currentEdt = *guid;
        edt->outputEvent = NULL_GUID;
        edt->depcNeeded = depc;

        if(paramc)
            memcpy((u64*) (edt+1), paramv, sizeof(u64) * paramc);

        if(eventGuid != NULL_GUID)
        {
            edt->outputEvent = eventGuid;
            hiveAddDependence(*guid, eventGuid, HIVE_EVENT_LATCH_DECR_SLOT, DB_MODE_NON_COHERENT_READ);
        }

        if(route != hiveGlobalRankId)
            hiveRemoteMemoryMove(route, *guid, (void*)edt, (unsigned int)edt->header.size, HIVE_REMOTE_EDT_MOVE_MSG, hiveFree);
        else
        {
            if(createdGuid) //this is a brand new edt
            {
                hiveRouteTableAddItem(edt, *guid, hiveGlobalRankId, false);
                if(edt->depcNeeded == 0)
                    hiveHandleReadyEdt((void*)edt);
            }
            else //we are racing to add an edt
            {
                hiveRouteTableAddItemRace(edt, *guid, hiveGlobalRankId, false);
                if(edt->depcNeeded)
                {
                    hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler); //Check the OO callback for EDT
                }
                else
                    hiveHandleReadyEdt((void*)edt);
            }
        }
        return true;
    }
    return false;
}

bool hiveEventCreateInternal(hiveGuid_t * guid, unsigned int route, hiveEventTypes_t eventType, unsigned int dependentCount, unsigned int latchCount, bool destroyOnFire)
{
    unsigned int eventSize = sizeof (struct hiveEvent) + sizeof (struct hiveDependent) * dependentCount;
    HIVESETMEMSHOTTYPE(hiveEventMemorySize);
    void * eventPacket = hiveCalloc(eventSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);

    if(eventSize)
    {
        struct hiveEvent *event = eventPacket;
        event->header.type = HIVE_EVENT;
        event->header.size = eventSize;
        event->eventType = eventType;
        event->dependentCount = 0;
        event->dependent.size = dependentCount;
        event->latchCount = latchCount;
        event->destroyOnFire = (destroyOnFire) ? dependentCount : -1;
        event->data = NULL_GUID;

        if(route == hiveGlobalRankId)
        {
            if(*guid)
            {
                hiveRouteTableAddItem(eventPacket, *guid, hiveGlobalRankId, false);
                hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler);
            }
            else
            {
                *guid = hiveGuidCreateForRank(route, HIVE_EVENT);
                hiveRouteTableAddItem(eventPacket, *guid, hiveGlobalRankId, false);
            }
        }
        else
            hiveRemoteMemoryMove(route, *guid, eventPacket, eventSize, HIVE_REMOTE_EVENT_MOVE_MSG, hiveFree);

        return true;
    }
    return false;
}

hiveGuid_t hiveEdtCreate(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depc * sizeof(hiveEdtDep_t);
    hiveGuid_t guid = NULL_GUID;
    hiveEdtCreateInternal(&guid, route, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

hiveGuid_t hiveEdtCreateWithGuid(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int route = hiveGuidGetRank(guid);
    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depc * sizeof(hiveEdtDep_t);
    bool ret = hiveEdtCreateInternal(&guid, route, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc);
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

hiveGuid_t hiveEdtCreateWithEvent(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc)
{
    HIVEEDTCOUNTERTIMERSTART(edtCreateCounter);
    hiveGuid_t guid = NULL_GUID;

    hiveGuid_t eventGuid = NULL_GUID;
    if(hiveEventCreateInternal(&eventGuid, route, HIVE_EVENT_LATCH_T, INITIAL_DEPENDENT_SIZE, 1, false))
    {
        unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depc * sizeof(hiveEdtDep_t);
        hiveEdtCreateInternal(&guid, route, edtSpace, eventGuid, funcPtr, paramc, paramv, depc);
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

void hiveEdtFree(struct hiveEdt * edt)
{
    hiveThreadInfo.edtFree = 1;
    hiveFree(edt);
    hiveThreadInfo.edtFree = 0;
}

inline void hiveEdtDelete(struct hiveEdt * edt)
{
    hiveRouteTableRemoveItem(edt->currentEdt);
    hiveEdtFree(edt);
}

void hiveEdtDestroy(hiveGuid_t guid)
{
    struct hiveEdt * edt = (struct hiveEdt*)hiveRouteTableLookupItem(guid);
    hiveRouteTableRemoveItem(guid);
    hiveEdtFree(edt);
}

hiveGuid_t hiveEventCreate(unsigned int route, hiveEventTypes_t eventType)
{
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    hiveGuid_t guid = NULL_GUID;
    hiveEventCreateInternal(&guid, route, eventType,  INITIAL_DEPENDENT_SIZE, 0, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return guid;
}

hiveGuid_t hiveEventCreateLatch(unsigned int route, unsigned int latchCount)
{
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    hiveGuid_t guid = NULL_GUID;
    hiveEventCreateInternal(&guid, route, HIVE_EVENT_LATCH_T,  INITIAL_DEPENDENT_SIZE, latchCount, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return guid;
}

hiveGuid_t hiveEventCreateWithGuid(hiveGuid_t guid, hiveEventTypes_t eventType)
{
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    unsigned int route = hiveGuidGetRank(guid);
    bool ret = hiveEventCreateInternal(&guid, route, eventType,  INITIAL_DEPENDENT_SIZE, 0, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

hiveGuid_t hiveEventCreateLatchWithGuid(hiveGuid_t guid, unsigned int latchCount)
{
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    unsigned int route = hiveGuidGetRank(guid);
    bool ret = hiveEventCreateInternal(&guid, route, HIVE_EVENT_LATCH_T,  INITIAL_DEPENDENT_SIZE, latchCount, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

void hiveEventFree(struct hiveEvent * event)
{
    struct hiveDependentList * trail, * current = event->dependent.next;
    while(current)
    {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
    hiveFree(event);
}

void hiveEventDestroy(hiveGuid_t guid)
{
    struct hiveEvent * event = (struct hiveEvent*)hiveRouteTableLookupItem( guid);
    if(event != NULL)
        hiveRouteTableRemoveItem(guid);
}

void hiveSignalEdt(hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveDbAccessMode_t mode)
{
    
    HIVEEDTCOUNTERTIMERSTART(signalEdtCounter);
    if(currentEdt && currentEdt->invalidateCount > 0)
        hiveOutOfOrderSignalEdt(currentEdt->currentEdt, edtPacket, dataGuid, slot, mode);
    else
    {
        struct hiveEdt * edt = hiveRouteTableLookupItem(edtPacket);
        if(edt == NULL)
        {
            unsigned int rank = hiveGuidGetRank(edtPacket);
            if(rank != hiveGlobalRankId)
                hiveRemoteSignalEdt(edtPacket, dataGuid, slot, mode);
            else
            {
                rank = hiveRouteTableLookupRank(edtPacket);
                if(rank == hiveGlobalRankId || rank == -1)
                    hiveOutOfOrderSignalEdt(edtPacket, edtPacket, dataGuid, slot, mode);
                else
                    hiveRemoteSignalEdt(edtPacket, dataGuid, slot, mode);
            }
        }
        else
        {
            hiveEdtDep_t *edtDep = (hiveEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
            if(slot < edt->depc)
            {
                edtDep[slot].guid = dataGuid;
                edtDep[slot].mode = mode;
            }
            unsigned int res = hiveAtomicSub(&edt->depcNeeded, 1U);
//            PRINTF("SIG: %lu %lu %u res: %u\n", edtPacket, dataGuid, slot, res);
            if(res == 0)
                hiveHandleReadyEdt(edt);
        }
    }
    hiveUpdatePerformanceMetric(hiveEdtSignalThroughput, hiveThread, 1, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(signalEdtCounter);
}

void hiveSignalEdtPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, u32 slot)
{
    struct hiveEdt * edt = hiveRouteTableLookupItem(edtGuid);
    if(edt)
    {
        hiveEdtDep_t *edtDep = (hiveEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
        
        if(slot < edt->depc)
        {
            edtDep[slot].guid = dbGuid;
            edtDep[slot].ptr = ptr;
            edtDep[slot].mode = DB_MODE_PTR;
        }
        if(hiveAtomicSub(&edt->depcNeeded, 1U) == 0)
            hiveHandleReadyEdt(edt);
    }
    else
    {
        unsigned int rank = hiveGuidGetRank(edtGuid);
        if(rank != hiveGlobalRankId)
        {
            hiveRemoteSignalEdtWithPtr(edtGuid, dbGuid, ptr, size, slot);
        }
        else
        {
            rank = hiveRouteTableLookupRank(edtGuid);
            if(rank == hiveGlobalRankId || rank == -1)
            {
                hiveOutOfOrderSignalEdtWithPtr(edtGuid, dbGuid, ptr, size, slot);
            }
            else
            {
                hiveRemoteSignalEdtWithPtr(edtGuid, dbGuid, ptr, size, slot);
            }
        }
    }
}

void hiveEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot)
{
    HIVEEDTCOUNTERTIMERSTART(signalEventCounter);
    if(currentEdt && currentEdt->invalidateCount > 0)
    {
        hiveOutOfOrderEventSatisfySlot(currentEdt->currentEdt, eventGuid, dataGuid, slot);
        return;
    }

    struct hiveEvent * event = (struct hiveEvent *) hiveRouteTableLookupItem(eventGuid);
    if(!event)
    {
        unsigned int rank = hiveGuidGetRank(eventGuid);
        if(rank != hiveGlobalRankId)
        {
            hiveRemoteEventSatisfySlot(eventGuid, dataGuid, slot);
        }
        else
        {
            hiveOutOfOrderEventSatisfySlot(eventGuid, eventGuid, dataGuid, slot);
        }
    }
    else if(event->eventType == HIVE_EVENT_LATCH_T)
    {
        if(event->fired)
        {
            PRINTF("HIVE_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
            hiveDebugGenerateSegFault();
        }

        unsigned int res;
        if(slot == HIVE_EVENT_LATCH_INCR_SLOT)
        {
            res = hiveAtomicAdd( &event->latchCount, 1U );
        }
        else if(slot == HIVE_EVENT_LATCH_DECR_SLOT)
        {
            if(dataGuid != NULL_GUID)
                event->data = dataGuid;
            res = hiveAtomicSub( &event->latchCount, 1U );
        }
        else
        {
            PRINTF("Bad latch slot %u\n", slot);
            hiveDebugGenerateSegFault();
        }

        if(!res)
        {
            if(hiveAtomicSwapBool(&event->fired, true))
            {
                PRINTF("HIVE_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
                hiveDebugGenerateSegFault();
            }
            else
            {
                struct hiveDependentList *dependentList = &event->dependent;
                struct hiveDependent *dependent = event->dependent.dependents;
                int i, j;
                unsigned int lastKnown = hiveAtomicFetchAdd(&event->dependentCount, 0U);
                event->pos = lastKnown + 1;
                i = 0;
                int totalSize = 0;
                while(i < lastKnown)
                {
                    j = i - totalSize;
                    while(i < lastKnown && j < dependentList->size)
                    {
                        while(!dependent[j].doneWriting);
                        if(dependent[j].type == HIVE_EDT)
                        {
                            hiveSignalEdt(dependent[j].addr, event->data, dependent[j].slot, dependent[j].mode);
                        }
                        else if (dependent[j].type == HIVE_EVENT)
                        {
                            #ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            hiveCounterTimerEndIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter));
                            uint64_t start = hiveCounterGetStartTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter));
                            #endif
                            hiveEventSatisfySlot(dependent[j].addr, event->data, dependent[j].slot);
                             #ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            hiveCounterSetEndTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter), start);
                            #endif
                        }
                        else if(dependent[j].type == HIVE_CALLBACK)
                        {
                            hiveEdtDep_t arg;
                            arg.guid = event->data;
                            arg.ptr = hiveRouteTableLookupItem(event->data);
                            arg.mode = DB_MODE_NON_COHERENT_READ;
                            dependent[j].callback(arg);
                        }
                        j++;
                        i++;
                    }
                    totalSize += dependentList->size;
                    while(i < lastKnown && dependentList->next == NULL);
                    dependentList = dependentList->next;
                    dependent = dependentList->dependents;
                }
                if(!event->destroyOnFire)
                {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(eventGuid);
                }
            }
        }
    }
    else
    {
        hiveEventSatisfy(eventGuid, dataGuid);
        //Don't update performance metrics since signalEvent will instead!
        return;
    }
    hiveUpdatePerformanceMetric(hiveEventSignalThroughput, hiveThread, 1, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(signalEventCounter);
}

void hiveEventSatisfy(hiveGuid_t eventGuid, hiveGuid_t dataGuid)
{
    HIVEEDTCOUNTERTIMERSTART(signalEventCounter);
    //Memory Model Stuff -> we delay until all writes are propogated
    if(currentEdt && currentEdt->invalidateCount > 0)
    {
        hiveOutOfOrderEventSatisfy(currentEdt->currentEdt, eventGuid, dataGuid);
        return;
    }

    void *eventAddress = hiveRouteTableLookupItem( eventGuid);
    if(eventAddress != NULL)
    {
        struct hiveEvent *event = eventAddress;
        bool isFired = false;
        if(event->eventType == HIVE_EVENT_IDEM_T)
            isFired = hiveAtomicSwapBool(&event->fired, true);
        else
            event->fired = true;
        if(!isFired)
        {
            struct hiveDependentList *dependentList = &event->dependent;
            struct hiveDependent *dependent = event->dependent.dependents;
            int i, j;
            unsigned int lastKnown = hiveAtomicFetchAdd(&event->dependentCount, 0U);
            event->pos = lastKnown + 1;
            event->data = dataGuid;
            i = 0;
            int totalSize = 0;
            while(i < lastKnown)
            {
                j = i - totalSize;
                while(i < lastKnown && j < dependentList->size)
                {
                    while(!dependent[j].doneWriting);
                    if(dependent[j].type == HIVE_EDT)
                    {
                        hiveSignalEdt(dependent[j].addr, dataGuid, dependent[j].slot, dependent[j].mode);
                    }
                    else if(dependent[j].type == HIVE_EVENT)
                    {
#ifdef COUNT
                        //THIS IS A TEMP FIX... problem is recursion...
                        hiveCounterTimerEndIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter));
                        uint64_t start = hiveCounterGetStartTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter));
#endif
                        hiveEventSatisfy(dependent[j].addr, dataGuid);
#ifdef COUNT
                        //THIS IS A TEMP FIX... problem is recursion...
                        hiveCounterSetEndTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid)?signalEventCounterOn:signalEventCounter), start);
#endif
                    }
                    else if(dependent[j].type == HIVE_CALLBACK)
                    {
                        hiveEdtDep_t arg;
                        arg.guid = event->data;
                        arg.ptr = hiveRouteTableLookupItem(event->data);
                        arg.mode = DB_MODE_NON_COHERENT_READ;
                        dependent[j].callback(arg);
                    }
                    j++;
                    i++;
                }
                totalSize += dependentList->size;
                while(i < lastKnown && dependentList->next == NULL);
                dependentList = dependentList->next;
                dependent = dependentList->dependents;
            }
            if(!event->destroyOnFire)
            {
                hiveEventFree(event);
                hiveRouteTableRemoveItem(eventGuid);
            }
        }
    }
    else
    {
        unsigned int rank = hiveGuidGetRank(eventGuid);
        if(rank != hiveGlobalRankId)
        {
            hiveRemoteEventSatisfy(eventGuid, dataGuid);
        }
        else
            hiveOutOfOrderEventSatisfy(eventGuid, eventGuid, dataGuid);
    }
    hiveUpdatePerformanceMetric(hiveEventSignalThroughput, hiveThread, 1, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(signalEventCounter);
}

struct hiveDependent * hiveDependentGet(struct hiveDependentList * head, int position)
{
    struct hiveDependentList * list = head;
    volatile struct hiveDependentList * temp;

    while(1)
    {
        //totalSize += list->size;

        if(position >= list->size)
        {
            if(position - list->size == 0)
            {
                if(list->next == NULL)
                {
                    temp = hiveCalloc(sizeof (struct hiveDependentList) + sizeof ( struct hiveDependent) *list->size * 2);
                    temp->size = list->size * 2;

                    list->next = (struct hiveDependentList *)temp;
                }
            }

            //EXPONENTIONAL BACK OFF THIS
            while(list->next == NULL)
            {
            }

            position -= list->size;
            list = list->next;
        }
        else
            break;
    }

    return list->dependents + position;
}

void hiveAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode)
{
    HIVEEDTCOUNTERTIMERSTART(addDependence);
    struct hiveHeader *sourceHeader = hiveRouteTableLookupItem(source);
    if(sourceHeader == NULL)
    {
        unsigned int rank = hiveGuidGetRank(source);
        if(rank != hiveGlobalRankId)
        {
            hiveRemoteAddDependence(source, destination, slot, mode, rank);
        }
        else
        {
            hiveOutOfOrderAddDependence(source, destination, slot, mode, source);
        }
        HIVEEDTCOUNTERTIMERENDINCREMENT(addDependence);
        return;
    }

    struct hiveEvent *event = (struct hiveEvent *)sourceHeader;
    if(hiveGuidGetType(destination) == HIVE_EDT)
    {
        struct hiveDependentList *dependentList = &event->dependent;
        struct hiveDependent *dependent;
        unsigned int position = hiveAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = hiveDependentGet(dependentList, position);
        dependent->type = HIVE_EDT;
        dependent->addr = destination;
        dependent->slot = slot;
        dependent->mode = mode;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? hiveAtomicSub(&event->destroyOnFire, 1U) : 1;
        if(event->fired)
        {
            while(event->pos == 0);
            if(position >= event->pos - 1)
            {
                hiveSignalEdt(destination, event->data, slot, mode);
                if(!destroyEvent)
                {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    }
    else if(hiveGuidGetType(destination) == HIVE_EVENT)
    {
        struct hiveDependentList *dependentList = &event->dependent;
        struct hiveDependent *dependent;
        unsigned int position = hiveAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = hiveDependentGet(dependentList, position);
        dependent->type = HIVE_EVENT;
        dependent->addr = destination;
        dependent->slot = slot;
        dependent->mode = mode;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? hiveAtomicSub(&event->destroyOnFire, 1U) : 1;
        if(event->fired)
        {
            while(event->pos == 0);
            if(event->pos - 1 <= position)
            {
                hiveEventSatisfySlot(destination, event->data, slot);
                if(!destroyEvent)
                {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(addDependence);
    return;
}

void hiveAddLocalEventCallback(hiveGuid_t source, eventCallback_t callback)
{
    HIVEEDTCOUNTERTIMERSTART(addDependence);
    struct hiveEvent *event = (struct hiveEvent *)hiveRouteTableLookupItem(source);
    if(event && hiveGuidGetType(source) == HIVE_EVENT)
    {
        struct hiveDependentList *dependentList = &event->dependent;
        struct hiveDependent *dependent;
        unsigned int position = hiveAtomicFetchAdd(&event->dependentCount, 1U);
        dependent = hiveDependentGet(dependentList, position);
        dependent->type = HIVE_CALLBACK;
        dependent->callback = callback;
        dependent->addr = NULL_GUID;
        dependent->slot = 0;
        COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
        dependent->doneWriting = true;

        unsigned int destroyEvent = (event->destroyOnFire != -1) ? hiveAtomicSub(&event->destroyOnFire, 1U) : 1;
        if(event->fired)
        {
            while(event->pos == 0);
            if(event->pos - 1 <= position)
            {
                hiveEdtDep_t arg;
                arg.guid = event->data;
                arg.ptr = hiveRouteTableLookupItem(event->data);
                arg.mode = DB_MODE_NON_COHERENT_READ;
                callback(arg);
                if(!destroyEvent)
                {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    }
    HIVEEDTCOUNTERTIMERENDINCREMENT(addDependence);
}

bool hiveIsEventFiredExt(hiveGuid_t event)
{
    bool fired = false;
    struct hiveEvent * actualEvent = (struct hiveEvent *)hiveRouteTableLookupItem(event);
    if(actualEvent)
        fired = actualEvent->fired;
    return fired;
}

//hiveGuid_t hivePercolateEdt(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t * depv)
//{
//    hiveGuid_t guid = NULL_GUID;
//    unsigned int edtSpace = sizeof(struct hiveEdt) + paramc * sizeof(u64) + depc * sizeof(hiveEdtDep_t);
//    if(route == hiveGlobalRankId)
//        hiveEdtCreateInternal(&guid, route, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, depv);
//    else
//    {
//        unsigned int dbSpace = 0;
//        for(unsigned int i=0; i<depc; i++)
//        {
//            unsigned int dbRank = hiveGuidGetRank(depv[i]);
//            if(dbRank == hiveGlobalRankId)
//            {
//                struct hiveHeader * dbHeader = hiveRouteTableLookupItem(depv[i]);
//                if(dbHeader)
//                {
//    //                PRINTF("dbGuid: %lu size: %u\n", depv[i], dbHeader->size);
//                    dbSpace+=dbHeader->size;
//                }
//                else
//                {
//                    //Do some out of order thing
//                }
//            }
//            else if(dbRank != route)
//            {
////                hiveRemoteDbForward(int pos, route, depv[i],  struct hiveEdt * edt, unsigned int type)
//                //Send some db request
//            }
//        }
//
//        struct hiveEdt *edt;
//        HIVESETMEMSHOTTYPE(hiveEdtMemorySize);
//        edt = (struct hiveEdt*)hiveCalloc(edtSpace + dbSpace);
//        HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);
//
//        if(edt)
//        {
//            guid = hiveGuidCreateForRank(route, HIVE_EDT);
//            edt->header.type = HIVE_EDT;
//            edt->header.size = edtSpace;
//            edt->funcPtr = funcPtr;
//            edt->depc = depc;
//            edt->paramc = paramc;
//            edt->freeMe = 1;
//            edt->currentEdt = guid;
//            edt->outputEvent = NULL_GUID;
//            edt->depcNeeded = depc;
//            edt->exclusiveWrite = false;
//            edt->writes = 0;
//            edt->fired = false;
//            edt->saveEwI = 0;
//            edt->ewSortList = NULL;
//
//            if(paramc)
//                memcpy((u64*) (edt+1), paramv, sizeof(u64) * paramc);
//
//            if(depv != NULL)
//            {
//                hiveEdtDep_t *edtDep = (hiveEdtDep_t *)((u64 *)(edt + 1) + edt->paramc);
//                char * next = (char *)(edtDep + depc);
//                for(unsigned int i=0; i<depc; i++)
//                {
//                    edtDep[i].guid = depv[i];
//                    edtDep[i].mode = DB_MODE_NON_COHERENT_READ;
//                    if(hiveGuidGetRank(depv[i]) == hiveGlobalRankId)
//                    {
//                        struct hiveHeader * dbHeader = hiveRouteTableLookupItem(depv[i]);
//                        if(dbHeader)
//                        {
//                            memcpy(next, dbHeader, dbHeader->size);
//                            next+=dbHeader->size;
//                        }
//                        else
//                        {
//                            //Do some out of order thing
//                        }
//                    }
//                    else
//                    {
//                        //Send some db request
//                    }
//                }
//            }
//            hiveActiveMessage(route, guid, edt, edtSpace + dbSpace);
//        }
//    }
//    return guid;
//}

hiveGuid_t hiveActiveMessageWithDb(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid)
{
    unsigned int rank = hiveGuidGetRank(dbGuid);
    hiveGuid_t guid = hiveEdtCreate(funcPtr, rank, paramc, paramv, depc+1);
//    PRINTF("AM -> %lu rank: %u depc: %u\n", guid, rank, depc+1);
    hiveSignalEdt(guid, dbGuid, 0, DB_MODE_PIN); //DB_MODE_NON_COHERENT_READ);
    return guid;
}

hiveGuid_t hiveActiveMessageWithBuffer(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * data, unsigned int size)
{
    void * ptr = hiveMalloc(size);
    memcpy(ptr, data, size);
    hiveGuid_t guid = hiveEdtCreate(funcPtr, route, paramc, paramv, depc+1);
    hiveSignalEdtPtr(guid, NULL_GUID, ptr, size, 0);
    return guid;
}
