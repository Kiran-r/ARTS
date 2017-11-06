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

inline void frontierLock(volatile unsigned int * lock)
{
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

inline void frontierUnlock(volatile unsigned int * lock)
{
    unsigned int mask = writeSet | exclusiveSet;
    hiveAtomicFetchAnd(lock, mask);
}

inline bool frontierAddReadLock(volatile unsigned int * lock)
{
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
inline bool frontierAddWriteLock(volatile unsigned int * lock)
{
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

inline bool frontierAddExclusiveLock(volatile unsigned int * lock)
{
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

inline void readerLock(volatile unsigned int * reader, volatile unsigned int * writer)
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

inline void readerUnlock(volatile unsigned int * reader)
{
    hiveAtomicSub(reader, 1U);
}

inline void writerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(hiveAtomicCswap(writer, 0U, 1U) == 0U);
    while(*reader);
    return;
}

inline bool nonBlockingWriteLock(volatile unsigned int * reader, volatile unsigned int * writer)
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

inline void writerUnlock(volatile unsigned int * writer)
{
    hiveAtomicSwap(writer, 0U);
}

inline struct hiveDbElement * hiveNewDbElement()
{
    struct hiveDbElement * ret = (struct hiveDbElement*) hiveCalloc(sizeof(struct hiveDbElement));  
    return ret;
}

inline struct hiveDbFrontier * hiveNewDbFrontier()
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

inline void hiveDeleteDbElement(struct hiveDbElement * head)
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

inline void hiveDeleteLocalDelayedEdt(struct hiveLocalDelayedEdt * head)
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

inline void hiveDeleteDbFrontier(struct hiveDbFrontier * frontier)
{
    if(frontier->list.next)
        hiveDeleteDbElement(frontier->list.next);
    if(frontier->localDelayed.next)
        hiveDeleteLocalDelayedEdt(frontier->localDelayed.next);
    hiveFree(frontier);
}

inline bool hivePushDbToElement(struct hiveDbElement * head, unsigned int position, unsigned int data)
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
}

inline void hivePushDelayedEdt(struct hiveLocalDelayedEdt * head, unsigned int position, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
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

inline bool hivePushDbToFrontier(struct hiveDbFrontier * frontier, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
{
    if(exclusive && !frontierAddExclusiveLock(&frontier->lock))
    {
        PRINTF("Failed hivePushDbToFrontier 1\n");
        return false;
    }
    else if(write && !frontierAddWriteLock(&frontier->lock))
    {
        PRINTF("Failed hivePushDbToFrontier 2\n");
        return false;        
    }
    else if(!exclusive && !write && !frontierAddReadLock(&frontier->lock))
    {
        PRINTF("Failed hivePushDbToFrontier 3\n");
        return false;
    }
    else if(bypass)
    {
        PRINTF("A\n");
        frontierLock(&frontier->lock);
        PRINTF("B\n");
    }
    if(hivePushDbToElement(&frontier->list, frontier->position, data))
        frontier->position++;
    
    if(exclusive)
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
bool hivePushDbToList(struct hiveDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode)
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
        PRINTF("FRONT: %p, W: %u E: %u L: %u B: %u %p\n", frontier, write, exclusive, local, bypass, edt);
        if(hivePushDbToFrontier(frontier, data, write, exclusive, local, bypass, edt, slot, mode))
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
        PRINTF("CLOSED LOCK %u %p\n", frontier->lock, frontier);
        iter = hiveDbFrontierIterCreate(frontier);

        frontierUnlock(&frontier->lock);
    }
    readerUnlock(&dbList->reader);
    PRINTF("CLOSED FRONTIER * %p\n", frontier);
    return iter;
    
}

inline void hiveSignalFrontierRemote(struct hiveDbFrontier * frontier, struct hiveDb * db, unsigned int getFrom)
{
    frontierLock(&frontier->lock);
    struct hiveDbFrontierIterator * iter = hiveDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(hiveDbFrontierIterNext(iter, &node))
        {
            if(node != hiveGlobalRankId)
            {
                hiveRemoteDbForward(node, getFrom, db->guid, DB_MODE_NON_COHERENT_READ); //Don't care about mode
            }                
        }
    }
    
    if(frontier->exNode)
    {
        //Send the exclusive here...
    }
    if(frontier->localPosition)
    {
        hiveRemoteDbForward(hiveGlobalRankId, getFrom, db->guid, DB_MODE_NON_COHERENT_READ); //Don't care about mode
        struct hiveOutOfOrderList * ooList = NULL;
        void ** data = hiveRouteTableGetOOList(db->guid, &ooList);
        struct hiveLocalDelayedEdt * current = &frontier->localDelayed;
        for(unsigned int i=0; i<frontier->localPosition; i++)
        {
            unsigned int pos = i % DBSPERELEMENT;
            struct hiveEdt * edt = current->edt[pos];
            unsigned int slot = current->slot[pos];
            hiveOutOfOrderHandleRemoteDbRequest(ooList, data, edt, slot);
           
            if(pos+1 == DBSPERELEMENT)
                current = current->next;
        }
    }
    frontierUnlock(&frontier->lock);
}

inline void hiveSignalFrontierLocal(struct hiveDbFrontier * frontier, struct hiveDb * db)
{
    frontierLock(&frontier->lock);
    struct hiveDbFrontierIterator * iter = hiveDbFrontierIterCreate(frontier);
    if(iter)
    {
        unsigned int node;
        while(hiveDbFrontierIterNext(iter, &node))
        {
            if(node != hiveGlobalRankId)
            {
                PRINTF("hiveSignalFrontierLocal -> hiveRemoteDbSendNow\n");
                hiveRemoteDbSendNow(node, db);
            }
        }
    }
    
    if(frontier->exNode)
    {
        //Send the exclusive here...
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

inline unsigned int * makeCopy(struct hiveDbFrontier * frontier)
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

inline void replaceWithCopy(struct hiveDbFrontier * frontier, unsigned int * array)
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

inline void sortSingleElement(struct hiveDbElement * current, unsigned int size)
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
