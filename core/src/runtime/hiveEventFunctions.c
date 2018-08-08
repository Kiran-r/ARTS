#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveGlobals.h"
#include "hiveAtomics.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveEdtFunctions.h"
#include "hiveEventFunctions.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveDebug.h"
#include <stdarg.h>
#include <string.h>

extern __thread struct hiveEdt * currentEdt;

bool hiveEventCreateInternal(hiveGuid_t * guid, unsigned int route, unsigned int dependentCount, unsigned int latchCount, bool destroyOnFire) {
    unsigned int eventSize = sizeof (struct hiveEvent) + sizeof (struct hiveDependent) * dependentCount;
    HIVESETMEMSHOTTYPE(hiveEventMemorySize);
    void * eventPacket = hiveCalloc(eventSize);
    HIVESETMEMSHOTTYPE(hiveDefaultMemorySize);

    if (eventSize) {
        struct hiveEvent *event = eventPacket;
        event->header.type = HIVE_EVENT;
        event->header.size = eventSize;
        event->dependentCount = 0;
        event->dependent.size = dependentCount;
        event->latchCount = latchCount;
        event->destroyOnFire = (destroyOnFire) ? dependentCount : -1;
        event->data = NULL_GUID;

        if (route == hiveGlobalRankId) {
            if (*guid) {
                hiveRouteTableAddItem(eventPacket, *guid, hiveGlobalRankId, false);
                hiveRouteTableFireOO(*guid, hiveOutOfOrderHandler);
            } else {
                *guid = hiveGuidCreateForRank(route, HIVE_EVENT);
                hiveRouteTableAddItem(eventPacket, *guid, hiveGlobalRankId, false);
            }
        } else
            hiveRemoteMemoryMove(route, *guid, eventPacket, eventSize, HIVE_REMOTE_EVENT_MOVE_MSG, hiveFree);

        return true;
    }
    return false;
}

hiveGuid_t hiveEventCreateLatch(unsigned int route, unsigned int latchCount) {
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    hiveGuid_t guid = NULL_GUID;
    hiveEventCreateInternal(&guid, route, INITIAL_DEPENDENT_SIZE, latchCount, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return guid;
}

hiveGuid_t hiveEventCreateLatchWithGuid(hiveGuid_t guid, unsigned int latchCount) {
    HIVEEDTCOUNTERTIMERSTART(eventCreateCounter);
    unsigned int route = hiveGuidGetRank(guid);
    bool ret = hiveEventCreateInternal(&guid, route, INITIAL_DEPENDENT_SIZE, latchCount, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(eventCreateCounter);
    return (ret) ? guid : NULL_GUID;
}

void hiveEventFree(struct hiveEvent * event) {
    struct hiveDependentList * trail, * current = event->dependent.next;
    while (current) {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
    hiveFree(event);
}

void hiveEventDestroy(hiveGuid_t guid) {
    struct hiveEvent * event = (struct hiveEvent*) hiveRouteTableLookupItem(guid);
    if (event != NULL)
    {
        hiveRouteTableRemoveItem(guid);
        hiveEventFree(event);
    }
}

void hiveEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot) {
    HIVEEDTCOUNTERTIMERSTART(signalEventCounter);
    if (currentEdt && currentEdt->invalidateCount > 0) {
        hiveOutOfOrderEventSatisfySlot(currentEdt->currentEdt, eventGuid, dataGuid, slot);
        return;
    }

    struct hiveEvent * event = (struct hiveEvent *) hiveRouteTableLookupItem(eventGuid);
    if (!event) {
        unsigned int rank = hiveGuidGetRank(eventGuid);
        if (rank != hiveGlobalRankId) {
            hiveRemoteEventSatisfySlot(eventGuid, dataGuid, slot);
        } else {
            hiveOutOfOrderEventSatisfySlot(eventGuid, eventGuid, dataGuid, slot);
        }
    } 
    else {
        if (event->fired) {
            PRINTF("HIVE_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
            hiveDebugGenerateSegFault();
        }

        unsigned int res;
        if (slot == HIVE_EVENT_LATCH_INCR_SLOT) {
            res = hiveAtomicAdd(&event->latchCount, 1U);
        } else if (slot == HIVE_EVENT_LATCH_DECR_SLOT) {
            if (dataGuid != NULL_GUID)
                event->data = dataGuid;
            res = hiveAtomicSub(&event->latchCount, 1U);
        } else {
            PRINTF("Bad latch slot %u\n", slot);
            hiveDebugGenerateSegFault();
        }

        if (!res) {
            if (hiveAtomicSwapBool(&event->fired, true)) {
                PRINTF("HIVE_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u\n", eventGuid, dataGuid, slot);
                hiveDebugGenerateSegFault();
            } else {
                struct hiveDependentList *dependentList = &event->dependent;
                struct hiveDependent *dependent = event->dependent.dependents;
                int i, j;
                unsigned int lastKnown = hiveAtomicFetchAdd(&event->dependentCount, 0U);
                event->pos = lastKnown + 1;
                i = 0;
                int totalSize = 0;
                while (i < lastKnown) {
                    j = i - totalSize;
                    while (i < lastKnown && j < dependentList->size) {
                        while (!dependent[j].doneWriting);
                        if (dependent[j].type == HIVE_EDT) {
                            hiveSignalEdt(dependent[j].addr, event->data, dependent[j].slot);
                        } else if (dependent[j].type == HIVE_EVENT) {
#ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            hiveCounterTimerEndIncrement(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter));
                            uint64_t start = hiveCounterGetStartTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter));
#endif
                            hiveEventSatisfySlot(dependent[j].addr, event->data, dependent[j].slot);
#ifdef COUNT
                            //THIS IS A TEMP FIX... problem is recursion...
                            hiveCounterSetEndTime(hiveGetCounter((hiveThreadInfo.currentEdtGuid) ? signalEventCounterOn : signalEventCounter), start);
#endif
                        } else if (dependent[j].type == HIVE_CALLBACK) {
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
                    while (i < lastKnown && dependentList->next == NULL);
                    dependentList = dependentList->next;
                    dependent = dependentList->dependents;
                }
                if (!event->destroyOnFire) {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(eventGuid);
                }
            }
        }
    }
    hiveUpdatePerformanceMetric(hiveEventSignalThroughput, hiveThread, 1, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(signalEventCounter);
}

struct hiveDependent * hiveDependentGet(struct hiveDependentList * head, int position) {
    struct hiveDependentList * list = head;
    volatile struct hiveDependentList * temp;

    while (1) {
        //totalSize += list->size;

        if (position >= list->size) {
            if (position - list->size == 0) {
                if (list->next == NULL) {
                    temp = hiveCalloc(sizeof (struct hiveDependentList) + sizeof ( struct hiveDependent) *list->size * 2);
                    temp->size = list->size * 2;

                    list->next = (struct hiveDependentList *) temp;
                }
            }

            //EXPONENTIONAL BACK OFF THIS
            while (list->next == NULL) {
            }

            position -= list->size;
            list = list->next;
        } else
            break;
    }

    return list->dependents + position;
}

void hiveAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot) {
    hiveDbAccessMode_t mode = DB_MODE_NON_COHERENT_READ;
    struct hiveHeader *sourceHeader = hiveRouteTableLookupItem(source);
    if (sourceHeader == NULL) {
        unsigned int rank = hiveGuidGetRank(source);
        if (rank != hiveGlobalRankId) {
            hiveRemoteAddDependence(source, destination, slot, mode, rank);
        } else {
            hiveOutOfOrderAddDependence(source, destination, slot, mode, source);
        }
        return;
    }

    struct hiveEvent *event = (struct hiveEvent *) sourceHeader;
    if (hiveGuidGetType(destination) == HIVE_EDT) {
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
        if (event->fired) {
            while (event->pos == 0);
            if (position >= event->pos - 1) {
                hiveSignalEdt(destination, event->data, slot);
                if (!destroyEvent) {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    } else if (hiveGuidGetType(destination) == HIVE_EVENT) {
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
        if (event->fired) {
            while (event->pos == 0);
            if (event->pos - 1 <= position) {
                hiveEventSatisfySlot(destination, event->data, slot);
                if (!destroyEvent) {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    }
    return;
}

void hiveAddLocalEventCallback(hiveGuid_t source, eventCallback_t callback) {
    struct hiveEvent *event = (struct hiveEvent *) hiveRouteTableLookupItem(source);
    if (event && hiveGuidGetType(source) == HIVE_EVENT) {
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
        if (event->fired) {
            while (event->pos == 0);
            if (event->pos - 1 <= position) {
                hiveEdtDep_t arg;
                arg.guid = event->data;
                arg.ptr = hiveRouteTableLookupItem(event->data);
                arg.mode = DB_MODE_NON_COHERENT_READ;
                callback(arg);
                if (!destroyEvent) {
                    hiveEventFree(event);
                    hiveRouteTableRemoveItem(source);
                }
            }
        }
    }
}

bool hiveIsEventFiredExt(hiveGuid_t event) {
    bool fired = false;
    struct hiveEvent * actualEvent = (struct hiveEvent *) hiveRouteTableLookupItem(event);
    if (actualEvent)
        fired = actualEvent->fired;
    return fired;
}