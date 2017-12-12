#include "hive.h"
#include "hiveAtomics.h"
#include "hiveMalloc.h"
#include "hiveOutOfOrderList.h"
#include "hiveDebug.h"

#define DPRINTF
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define fireLock 1U
#define resetLock 2U

bool readerOOTryLock(struct hiveOutOfOrderList * list)
{
    while(1)
    {
        if(list->writerLock == fireLock)
            return false;
        while(list->writerLock == resetLock);
        hiveAtomicFetchAdd(&list->readerLock, 1U);
        if(list->writerLock==0)
            break;
        hiveAtomicSub(&list->readerLock, 1U);
    }
    return true;
}

inline void readerOOLock(struct hiveOutOfOrderList *  list)
{
    while(1)
    {
        while(list->writerLock);
        hiveAtomicFetchAdd(&list->readerLock, 1U);
        if(list->writerLock==0)
            break;
        hiveAtomicSub(&list->readerLock, 1U);
    }
}

void readerOOUnlock(struct hiveOutOfOrderList *  list)
{
    hiveAtomicSub(&list->readerLock, 1U);
}

void writerOOLock(struct hiveOutOfOrderList *  list, unsigned int lockType)
{
    while(hiveAtomicCswap(&list->writerLock, 0U, lockType) == 0U);
    while(list->readerLock);
    return;
}

bool writerTryOOLock(struct hiveOutOfOrderList *  list, unsigned int lockType)
{
    while(1)
    {
        unsigned int temp = hiveAtomicCswap(&list->writerLock, 0U, lockType);
        if(temp == 0U)
        {
            while(list->readerLock);
            break;
        }
        if(temp == lockType)
            return false;
    }
    return true;
}

void writerOOUnlock(struct hiveOutOfOrderList *  list)
{
    hiveAtomicSwap(&list->writerLock, 0U);
}

bool hiveOOisFired(struct hiveOutOfOrderList *  list)
{
    return list->isFired;
}

bool hiveOutOfOrderListAddItem(struct hiveOutOfOrderList * addToMe, void * item)
{
    if(!readerOOTryLock(addToMe))
    {
        return false;
    }
    if(hiveOOisFired(addToMe))
    {
        readerOOUnlock(addToMe);
        return false;
    }
    unsigned int pos = hiveAtomicFetchAdd(&addToMe->count, 1U);

    DPRINTF("ADDING to OO LIST %u %u %p\n", pos, addToMe->count, &addToMe->count);
    unsigned int numElements = pos / OOPERELEMENT;
    unsigned int elementPos = pos % OOPERELEMENT;

    volatile struct hiveOutOfOrderElement * current = &addToMe->head;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(!current->next)
        {
            if(i+1 == numElements && elementPos == 0 )
            {
                current->next = hiveCalloc(sizeof(struct hiveOutOfOrderElement));
            }
            else
                while(!current->next);
        }
        current=current->next;
    }
    if(hiveAtomicCswapPtr((volatile void**)&current->array[elementPos], (void*)0, item))
        PRINTF("OO pos not empty...\n");
    readerOOUnlock(addToMe);
    return true;
}

void hiveOutOfOrderListReset(struct hiveOutOfOrderList * list)
{
    if(writerTryOOLock(list, resetLock))
    {
        list->isFired = false;
        if(list->count)
        {
            PRINTF("Reseting but OO is not empty\n");
        }
        writerOOUnlock(list);
    }
}

void deleteOOElements(struct hiveOutOfOrderElement * current)
{
    struct hiveOutOfOrderElement * trail = NULL;
    while(current)
    {
        for(unsigned int i=0; i<OOPERELEMENT; i++)
        {
            while(current->array[i]);
        }
        trail = current;
        current = (struct hiveOutOfOrderElement *) current->next;
        hiveFree(trail);
    }
}

//Not threadsafe
void hiveOutOfOrderListDelete(struct hiveOutOfOrderList * list)
{
    deleteOOElements((struct hiveOutOfOrderElement *) list->head.next);
    list->head.next = NULL;
    list->isFired = false;
    list->count = 0;
}

void hiveOutOfOrderListFireCallback(struct hiveOutOfOrderList * fireMe, void * localGuidAddress,  void (* callback )( void *, void * ))
{
    if(writerTryOOLock(fireMe, fireLock))
    {
        DPRINTF("FIRING OO LIST %u\n", fireMe->count);
        fireMe->isFired = true;
        unsigned int pos = fireMe->count;
        unsigned int j = 0;
        for(volatile struct hiveOutOfOrderElement * current=&fireMe->head; current; current=current->next)
        {
            for(unsigned int i=0; i<OOPERELEMENT; i++)
            {
                if(j<pos)
                {
                    volatile void * item = NULL;
                    while(!item)
                    {
                        item = hiveAtomicSwapPtr((volatile void *)&current->array[i], (void*)0);
                    }
                    callback((void *)item, localGuidAddress);
                    j++;
                }
            }
            if(j == pos)
                break;
            while(!current->next);
        }
        fireMe->count = 0;
        struct hiveOutOfOrderElement * p = (struct hiveOutOfOrderElement *) fireMe->head.next;
        fireMe->head.next = NULL;
        writerOOUnlock(fireMe);
        deleteOOElements(p);
    }
}
