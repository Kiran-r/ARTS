#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hiveGlobals.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveDbList.h"
#include "hiveRemoteFunctions.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveRuntime.h"
#include "stdint.h"
#include "inttypes.h"

#define writeSet     0x80000000
#define exclusiveSet 0x40000000

#define DPRINTF(...)

void frontierLock(volatile unsigned int * lock)
{
    DPRINTF("Llocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & 1U) == 0)
        {
            temp = hiveAtomicCswap(lock, local, local | 1U);
            if(temp == local)
            {
                return;
            }
        }
    }
}

void frontierUnlock(volatile unsigned int * lock)
{
    DPRINTF("unlocking frontier: <<<<<<< %p\n", lock);
    unsigned int mask = writeSet | exclusiveSet;
    hiveAtomicFetchAnd(lock, mask);
}

bool frontierAddReadLock(volatile unsigned int * lock)
{
    DPRINTF("Rlocking frontier: >>>>>>> %p %u\n", lock, *lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & 1U) == 0)
        {
            temp = hiveAtomicCswap(lock, local, local | 1U);
            if(temp == local)
            {
                return true;
            }
        }
    }
}

//Returns true if there is no write in the frontier, false if there is
bool frontierAddWriteLock(volatile unsigned int * lock)
{
    DPRINTF("Wlocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & writeSet) != 0) //Make sure write not set first
            return false;
        if((local & 1U) == 0) //Wait for lock to be free
        {
            temp = hiveAtomicCswap(lock, local, local | writeSet | 1U);
            if(temp == local)
            {
                return true;
            }
        }
    }
}

bool frontierAddExclusiveLock(volatile unsigned int * lock)
{
    DPRINTF("Elocking frontier: >>>>>>> %p\n", lock);
    unsigned int local, temp;
    while(1)
    {
        local = *lock;
        if((local & exclusiveSet) != 0) //Make sure exclusive not set first
            return false;
        if((local & writeSet) != 0) //Make sure write not set first
            return false;
        if((local & 1U) == 0) //We reserved the write, now wait for lock to be free
        {
            temp = hiveAtomicCswap(lock, local, local | exclusiveSet | writeSet | 1U);
            if(temp == local)
            {
                return true;
            }
        }
        return true;
    }
}

void readerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(1)
    {
        while(*writer);
        hiveAtomicFetchAdd(reader, 1U);
        if(*writer==0)
            break;
        hiveAtomicSub(reader, 1U);
    }
}

void readerUnlock(volatile unsigned int * reader)
{
    hiveAtomicSub(reader, 1U);
}

void writerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(hiveAtomicCswap(writer, 0U, 1U) == 0U);
    while(*reader);
    return;
}

bool nonBlockingWriteLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    if(hiveAtomicCswap(writer, 0U, 1U) == 0U)
    {
        while(*reader);
        return true;
    }
    else
    {
        return false;
    }
}

void writerUnlock(volatile unsigned int * writer)
{
    hiveAtomicSwap(writer, 0U);
}

struct hiveDbElement * hiveNewDbElement()
{
    struct hiveDbElement * ret = (struct hiveDbElement*) hiveCalloc(sizeof(struct hiveDbElement));
    return ret;
}

struct hiveDbFrontier * hiveNewDbFrontier()
{
    struct hiveDbFrontier * ret = (struct hiveDbFrontier*) hiveCalloc(sizeof(struct hiveDbFrontier));
    return ret;
}

//This should be done before being released into the wild
struct hiveDbList * hiveNewDbList()
{
    struct hiveDbList * ret = (struct hiveDbList *) hiveCalloc(sizeof(struct hiveDbList));
    ret->head = ret->tail = hiveNewDbFrontier();
    return ret;
}

void hiveDeleteDbElement(struct hiveDbElement * head)
{
    struct hiveDbElement * trail;
    struct hiveDbElement * current;
    while(current)
    {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
}

void hiveDeleteLocalDelayedEdt(struct hiveLocalDelayedEdt * head)
{
    struct hiveLocalDelayedEdt * trail;
    struct hiveLocalDelayedEdt * current;
    while(current)
    {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
}

void hiveDeleteDbFrontier(struct hiveDbFrontier * frontier)
{
    if(frontier->list.next)
        hiveDeleteDbElement(frontier->list.next);
    if(frontier->localDelayed.next)
        hiveDeleteLocalDelayedEdt(frontier->localDelayed.next);
    hiveFree(frontier);
}

bool hivePushDbToElement(struct hiveDbElement * head, unsigned int position, unsigned int data)
{
    unsigned int j = 0;
    for(struct hiveDbElement * current=head; current; current=current->next)
    {
        for(unsigned int i=0; i<DBSPERELEMENT; i++)
        {
            if(j<position)
            {
                if(current->array[i] == data)
                {
                    return false;
                }
                j++;
            }
            else
            {
                current->array[i] = data;
                return true;
            }
        }
        if(!current->next)
            current->next = hiveNewDbElement();
    }
    // Need to mark unreachable
    return false;
}

void hivePushDelayedEdt(struct hiveLocalDelayedEdt * head, unsigned int position, struct hiveEdt * edt, unsigned int slot, hiveType_t mode)
{
    unsigned int numElements = position / DBSPERELEMENT;
    unsigned int elementPos = position % DBSPERELEMENT;
    struct hiveLocalDelayedEdt * current = head;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(!current->next)
            current->next = hiveMalloc(sizeof(struct hiveLocalDelayedEdt));
        current=current->next;
    }
    current->edt[elementPos] = edt;
    current->slot[elementPos] = slot;
    current->mode[elementPos] = mode;
}

bool hivePushDbToFrontier(struct hiveDbFrontier * frontier, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct hiveEdt * edt, unsigned int slot, hiveType_t mode, bool * unique)
{
    if(bypass)
    {
        DPRINTF("A\n");
        frontierLock(&frontier->lock);
        DPRINTF("B\n");
    }
    else if(exclusive && !frontierAddExclusiveLock(&frontier->lock))
    {
        DPRINTF("Failed hivePushDbToFrontier 1\n");
        return false;
    }
    else if(write && !frontierAddWriteLock(&frontier->lock))
    {
        DPRINTF("Failed hivePushDbToFrontier 2\n");
        return false;
    }
    else if(!exclusive && !write && !frontierAddReadLock(&frontier->lock))
    {
        DPRINTF("Failed hivePushDbToFrontier 3\n");
        return false;
    }

    if(hivePushDbToElement(&frontier->list, frontier->position, data))
        frontier->position++;
//    else
//    {
//        if(*unique && !local)
//            PRINTF("agging the write after read\n");
//        *unique = local;
//
//    }

    DPRINTF("*unique %u w: %u e: %u l: %u b: %u rank: %u\n", *unique, write, exclusive, local, bypass, data);

    if(exclusive || (write && !local))
    {
        frontier->exNode = data;
        frontier->exEdt = edt;
        frontier->exSlot = slot;
        frontier->exMode = mode;
    }
    else if(local)
    {
        hivePushDelayedEdt(&frontier->localDelayed, frontier->localPosition++, edt, slot, mode);
    }

    frontierUnlock(&frontier->lock);
    return true;
}

//Returns if the push is to the head frontier
/* A read after write from the same node would send duplicate copies of DB.
 * To fix this, if the node is remote, we only return true if the adding the
 * rank to the frontier is unique.  If the db is local then we return if the DB
 * is added to the first frontier reguardless of if there are duplicates.
 */
bool hivePushDbToList(struct hiveDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct hiveEdt * edt, unsigned int slot, hiveType_t mode)
{
    if(!dbList->head)
    {
        if(nonBlockingWriteLock(&dbList->reader, &dbList->writer))
        {
            dbList->head = dbList->tail = hiveNewDbFrontier();
            writerUnlock(&dbList->writer);
        }
    }
    readerLock(&dbList->reader, &dbList->writer);
    struct hiveDbFrontier * frontier = dbList->head;
    bool ret = true;
    for(struct hiveDbFrontier * frontier = dbList->head; frontier; frontier=frontier->next)
    {
        DPRINTF("FRONT: %p, W: %u E: %u L: %u B: %u %p\n", frontier, write, exclusive, local, bypass, edt);
        if(hivePushDbToFrontier(frontier, data, write, exclusive, local, bypass, edt, slot, mode, &ret))
            break;
        else
        {
            if(!frontier->next)
            {
                struct hiveDbFrontier * newFrontier = hiveNewDbFrontier();
                if(hiveAtomicCswapPtr((void*)&frontier->next, NULL, newFrontier))
                {
                    hiveDeleteDbFrontier(newFrontier);
                    while(!frontier->next);
                }
            }
        }
        ret = false;
    }
    readerUnlock(&dbList->reader);
    return ret;
}

unsigned int hiveCurrentFrontierSize(struct hiveDbList * dbList)
{
    unsigned int size;
    readerLock(&dbList->reader, &dbList->writer);
    if(dbList->head)
    {
        frontierLock(&dbList->head->lock);
        size = dbList->head->position;
        frontierUnlock(&dbList->head->lock);
    }
    readerUnlock(&dbList->head->lock);
    return size;
}

struct hiveDbFrontierIterator * hiveDbFrontierIterCreate(struct hiveDbFrontier * frontier)
{
    struct hiveDbFrontierIterator * iter = NULL;
    if(frontier && frontier->position)
    {
        iter = (struct hiveDbFrontierIterator *) hiveCalloc(sizeof(struct hiveDbFrontierIterator));
        iter->frontier = frontier;
        iter->currentElement = &frontier->list;
    }
    // Need to mark unreachable
    return NULL;
}

unsigned int hiveDbFrontierIterSize(struct hiveDbFrontierIterator * iter)
{
    return iter->frontier->position;
}

bool hiveDbFrontierIterNext(struct hiveDbFrontierIterator * iter, unsigned int * next)
{
    if(iter->currentIndex < iter->frontier->position)
    {
        *next = iter->currentElement->array[iter->currentIndex++ % DBSPERELEMENT];
        if(!(iter->currentIndex % DBSPERELEMENT))
        {
            iter->currentElement = iter->currentElement->next;
        }
        return true;
    }
    return false;
}

bool hiveDbFrontierIterHasNext(struct hiveDbFrontierIterator * iter)
{
    return (iter->currentIndex < iter->frontier->position);
}

void hiveDbFrontierIterDelete(struct hiveDbFrontierIterator * iter)
{
    hiveFree(iter->frontier);
    hiveFree(iter);
}

struct hiveDbFrontierIterator * hiveCloseFrontier(struct hiveDbList * dbList)
{
    struct hiveDbFrontierIterator * iter = NULL;
    readerLock(&dbList->reader, &dbList->writer);
    struct hiveDbFrontier * frontier = dbList->head;
    if(frontier)
    {
        frontierLock(&frontier->lock);

        hiveAtomicFetchOr(&frontier->lock, exclusiveSet | writeSet | 1U);
        iter = hiveDbFrontierIterCreate(frontier);

        frontierUnlock(&frontier->lock);
    }
    readerUnlock(&dbList->reader);
    return iter;

}

void hiveSignalFrontierRemote(struct hiveDbFrontier * frontier, struct hiveDb * db, unsigned int getFrom)
{
    frontierLock(&frontier->lock);

    if(frontier->exEdt)
    {
        if(frontier->exNode == getFrom)
            hiveRemoteSendAlreadyLocal(getFrom, db->guid, frontier->exEdt, frontier->exSlot, frontier->exMode);
        else if(frontier->exNode != hiveGlobalRankId)
            hiveRemoteDbForwardFull(frontier->exNode, getFrom, db->guid, frontier->exEdt, frontier->exSlot, frontier->exMode);
        else
            hiveRemoteDbFullRequest(db->guid, getFrom, frontier->exEdt, frontier->exSlot, frontier->exMode);

    }

    struct hiveDbFrontierIterator * iter = hiveDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(hiveDbFrontierIterNext(iter, &node))
        {
            if(node != hiveGlobalRankId && !(frontier->exEdt &&  node == frontier->exNode))
            {
                hiveRemoteDbForward(node, getFrom, db->guid, HIVE_DB_READ); //Don't care about mode
            }
        }
    }

    if(frontier->localPosition)
    {
        struct hiveLocalDelayedEdt * current = &frontier->localDelayed;
        for(unsigned int i=0; i<frontier->localPosition; i++)
        {
            unsigned int pos = i % DBSPERELEMENT;
            struct hiveEdt * edt = current->edt[pos];
            unsigned int slot = current->slot[pos];
            //send through aggregation
            hiveRemoteDbRequest(db->guid, getFrom, edt, slot, HIVE_DB_READ, true);
            if(pos+1 == DBSPERELEMENT)
                current = current->next;
        }
    }

    if(hivePushDbToElement(&frontier->list, frontier->position, getFrom))
        frontier->position++;
    frontierUnlock(&frontier->lock);
}

void hiveSignalFrontierLocal(struct hiveDbFrontier * frontier, struct hiveDb * db)
{
    frontierLock(&frontier->lock);

    if(frontier->exEdt)
    {
        if(frontier->exNode == hiveGlobalRankId)
        {
            struct hiveEdt * edt = frontier->exEdt;
            hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
            depv[frontier->exSlot].ptr = db+1;
            if(hiveAtomicSub(&edt->depcNeeded,1U) == 0)
                hiveHandleRemoteStolenEdt(edt);
        }
        else
            hiveRemoteDbFullSendNow(frontier->exNode, db, frontier->exEdt, frontier->exSlot, frontier->exMode);
    }

    struct hiveDbFrontierIterator * iter = hiveDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(hiveDbFrontierIterNext(iter, &node))
        {
            if(node != hiveGlobalRankId && !(frontier->exEdt &&  node == frontier->exNode))
            {
                hiveRemoteDbSendNow(node, db);
                PRINTF("Progress Local sending to %u\n", node);
            }
        }
    }


    if(frontier->localPosition)
    {
        struct hiveLocalDelayedEdt * current = &frontier->localDelayed;
        for(unsigned int i=0; i<frontier->localPosition; i++)
        {
            unsigned int pos = i % DBSPERELEMENT;
            struct hiveEdt * edt = current->edt[pos];
            hiveEdtDep_t * depv = (hiveEdtDep_t *)(((u64 *)(edt + 1)) + edt->paramc);
            depv[current->slot[pos]].ptr = db+1;

            if(hiveAtomicSub(&edt->depcNeeded,1U) == 0)
            {
                hiveHandleRemoteStolenEdt(edt);
            }

            if(pos+1 == DBSPERELEMENT)
                current = current->next;
        }
    }
    frontierUnlock(&frontier->lock);
}

void hiveProgressFrontier(struct hiveDb * db, unsigned int rank)
{
    struct hiveDbList * dbList = db->dbList;
    writerLock(&dbList->reader, &dbList->writer);
    struct hiveDbFrontier * tail = dbList->head;
    if(dbList->head)
    {
        PRINTF("HEAD: %p -> NEXT: %p\n", dbList->head, dbList->head->next);
        dbList->head = (struct hiveDbFrontier *) dbList->head->next;
        if(dbList->head)
        {
            if(rank==hiveGlobalRankId)
                hiveSignalFrontierLocal(dbList->head, db);
            else
                hiveSignalFrontierRemote(dbList->head, db, rank);
        }
    }
    writerUnlock(&dbList->writer);
    //This should be safe since the writer lock ensures all readers are done
    if(tail)
        hiveDeleteDbFrontier(tail);
}

struct hiveDbFrontierIterator * hiveProgressAndGetFrontier(struct hiveDbList * dbList)
{
    writerLock(&dbList->reader, &dbList->writer);
    struct hiveDbFrontier * tail = dbList->head;
    dbList->head = (struct hiveDbFrontier *) dbList->head->next;
    writerUnlock(&dbList->writer);
    //This should be safe since the writer lock ensures all readers are done
    return hiveDbFrontierIterCreate(tail);
}

/*Not sure if we need then...**************************************************/

unsigned int * makeCopy(struct hiveDbFrontier * frontier)
{
    unsigned int * array = hiveMalloc(sizeof(unsigned int) * frontier->position);
    unsigned int numElements = frontier->position / DBSPERELEMENT;
    unsigned int lastPos = frontier->position % DBSPERELEMENT;
    struct hiveDbElement * current = &frontier->list;
    unsigned int k = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
        unsigned int bound = (i+1 == numElements) ? lastPos : DBSPERELEMENT;
        for(unsigned int j=0; j<bound; j++)
        {
            array[k++] = current->array[j];
        }
        current = current->next;
    }
    return array;
}

void quicksort(unsigned int * array, unsigned int length)
{
  if (length < 2)
      return;

  unsigned int pivot = array[length/2];
  unsigned int i, j;
  for(i = 0, j = length-1; ; i++, j--)
  {
    while(array[i] < pivot)
        i++;
    while(array[j] > pivot)
        j--;

    if (i >= j) break;
    unsigned int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }

  quicksort(array, i);
  quicksort(array+i, length-i);
}

void replaceWithCopy(struct hiveDbFrontier * frontier, unsigned int * array)
{
    unsigned int numElements = frontier->position / DBSPERELEMENT;
    unsigned int lastPos = frontier->position % DBSPERELEMENT;
    struct hiveDbElement * current = &frontier->list;
    unsigned int k = 0;
    for(unsigned int i=0; i<numElements; i++)
    {
        unsigned int bound = (i+1 == numElements) ? lastPos : DBSPERELEMENT;
        for(unsigned int j=0; j<bound; j++)
        {
            current->array[j] = array[k++];
        }
        current = current->next;
    }
}

void sortSingleElement(struct hiveDbElement * current, unsigned int size)
{
    unsigned int * array = current->array;
    for(unsigned int i=1; i<size; i++)
    {
        unsigned int temp = array[i];
        unsigned int j = i;
        while(j>0 && temp < array[j-1])
        {
            array[j] = array[j-1];
            j--;
        }
        array[j] = temp;
    }
}

void hiveDbFrontierSort(struct hiveDbFrontier * frontier)
{
    frontierLock(&frontier->lock);
    if(frontier->position > 1)
    {
        if(frontier->position <= DBSPERELEMENT)
            sortSingleElement(&frontier->list, frontier->position);
        else
        {
            unsigned int * copy = makeCopy(frontier);
            quicksort(copy, frontier->position);
            replaceWithCopy(frontier, copy);
            hiveFree(copy);
        }
    }
    frontierUnlock(&frontier->lock);
}
