#include "hive.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveDbLockList.h"
#include <string.h>

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

struct hiveDbLockListItem
{
    bool shared;
    unsigned int rank;
    volatile unsigned int count;
    volatile struct hiveDbLockListItem * volatile right;
    volatile struct hiveDbLockListItem * volatile end;
    volatile struct hiveDbLockListItem * volatile next;
    volatile void * volatile  data;
} __attribute__ ((aligned));

struct hiveDbLockList
{
    //unsigned int headLock;
    volatile struct hiveDbLockListItem * volatile headPtr;
    volatile struct hiveDbLockListItem * volatile tailPtr;
    volatile unsigned int tailLock;
    volatile unsigned int firstInLock;
} __attribute__ ((aligned));

void hiveDbLockListNew(struct hiveDbLockList *list)
{
    struct hiveDbLockListItem * newItem = hiveMalloc( sizeof(struct hiveDbLockListItem) );
    newItem->next=NULL;
    newItem->right=(void *) 0x1;
    newItem->shared=false;
    list->firstInLock = list->tailLock = 0U;
    //list->insertLock = 0U;
    list->headPtr = list->tailPtr = newItem;
}

struct hiveDbLockList * hiveDbLockListGroupNew(unsigned int listSize)
{
    DPRINTF("%d\n", sizeof( struct hiveDbLockListItem ));
    int i;
    struct hiveDbLockList *linkList =
        (struct hiveDbLockList *) hiveCalloc(sizeof (struct hiveDbLockList) * listSize);

    for (i = 0; i < listSize; i++)
        hiveDbLockListNew(linkList+i);

    return linkList;
}

inline struct hiveDbLockList * hiveDbLockListGet(struct hiveDbLockList *linkList, unsigned int position)
{
    return (struct hiveDbLockList *)(linkList + position);
}

void hiveDbLockListDelete(void *linkList)
{
    struct hiveDbLockList * list = linkList;
    struct hiveDbLockListItem * last;
    while(list->headPtr != NULL)
    {
        last = (struct hiveDbLockListItem *)list->headPtr;
        list->headPtr = list->headPtr->next;
        hiveFree(last);
    }
    hiveFree(linkList);
}

//Multiple Pushers
static bool insertRight(volatile struct hiveDbLockListItem * volatile tail, volatile struct hiveDbLockListItem * volatile item )
{
    void * res;
    volatile struct hiveDbLockListItem * volatile next; 
    next = tail->end;
    item->count = next->count+1;
    while(next != (void *) 0x1)
    {
        if(next->right == NULL)
        {
            res = (void *)hiveAtomicCswapPtr((volatile void **)&next->right, NULL, (void *)item);
            if(res == NULL)
            {
                //tail->count++;
                tail->end = item;
                break;
            }
        }
        next = next->right;
    }
    if(next != (void *)0x1)
        return true;
    item->count = 1;
    return false;
}

bool hiveDbLockListPush(struct hiveDbLockList *list, void *item, unsigned int rank, bool shared)
{
    volatile struct hiveDbLockListItem * volatile newItem = hiveMalloc( sizeof(struct hiveDbLockListItem) );
    newItem->next=NULL;
    newItem->data = item;
    newItem->right=NULL;
    newItem->rank = rank;
    newItem->end=newItem;
    newItem->count=1;
    newItem->shared = shared;

    volatile struct hiveDbLockListItem * volatile tail;
    volatile unsigned int res;
    while (1)
    {
        if(list->tailLock == 0U)
        {
            res = hiveAtomicCswap(&list->tailLock, 0U, 1U);
            if(res == 0U)
            {
                tail = list->tailPtr;
                while(1)
                {
                    if( list->firstInLock == 0U )
                    {
                            list->firstInLock = 1U;
                            list->tailLock = 0U;
                            hiveFree((void *)newItem);
                            return true;
                    }
                    else
                    {
                        if(shared && tail->shared)
                        {
                            if(insertRight(tail, newItem))
                            {
                                list->tailLock = 0U;
                                return false;
                            }
                            else
                            {
                                tail->next = newItem;
                                list->tailPtr = newItem;
                                COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
                                list->tailLock = 0U;
                                return false;
                            }
                        }
                        else
                        {
                            tail->next = newItem;
                            list->tailPtr = newItem;
                            COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
                            list->tailLock = 0U;
                            return false;
                        } 
                    }
                }
            }
        }
    }
}

static unsigned int popRight(volatile struct hiveDbLockListItem * volatile tail)
{
    void * res;
    unsigned int count;
    tail = tail->end;
    while(tail != (void *) 0x1)
    {

        if(tail->right == NULL)
        {
            res = (void *)hiveAtomicCswapPtr((volatile void **)&tail->right, NULL, (void *)0x1);

            if(res == NULL)
                break;
        }
        
        tail = tail->right;
    }

    return tail->count;
}

//Only 1 popper
static __thread void ** returnList = NULL;
static __thread unsigned int * returnRankList = NULL;
static __thread unsigned int returnSize=0;

void ** hiveDbLockListPop( struct hiveDbLockList * list, unsigned int ** rankList, unsigned int * size)
{
    volatile struct hiveDbLockListItem * volatile head;
    volatile struct hiveDbLockListItem * volatile tail;
    volatile struct hiveDbLockListItem * volatile backup;
    volatile void * volatile data;
    void * ptr;
    head = list->headPtr;
    volatile unsigned int res;
    while(1)
    {
        if( head->next != NULL )
        {
            //PRINTF("head %p\n", list->headPtr);
            
            list->headPtr = head->next;
            int count = popRight(head->next);
            
            if(count > returnSize)
            {
                hiveFree(returnList);
                hiveFree(returnRankList);
                returnList = hiveMalloc( sizeof(void *) * count );
                returnRankList = hiveMalloc( sizeof(unsigned int) * count );
            }
            backup = head;
            head = head->next;
            for(int i=0; i< count;i++)
            {
                returnList[i] = (void*)head->data;
                returnRankList[i] = head->rank;
                tail = head;
                head=head->right;
                //hiveFree((void *)tail);
            }
            head = backup;  
            while(head != (void *) 0x1) 
            {
                tail = head;
                head=head->right;
                hiveFree((void *)tail);
            }
            //PRINTF("%p\n",data); 
            *size = count;
            *rankList = returnRankList;
            return returnList;
        }
        else
        {
            res = hiveAtomicCswap(&list->tailLock, 0U, 1U);
            if(res == 0U)
            {
                if(head->next == NULL)
                {
                    list->firstInLock = 0U;
                    list->tailLock = 0U;
                    //PRINTF("Here\n");
                    *size=0;
                    return NULL;   
                }
                else
                {
                    list->tailLock = 0U;
                }
            }
        }
    }
}

void * hiveDbLockListPeekFront( struct hiveDbLockList * list)
{
    volatile struct hiveDbLockListItem * volatile head;
    volatile struct hiveDbLockListItem * volatile tail;
    volatile void * volatile data;
    void * ptr;
    head = list->headPtr;
    volatile unsigned int res;
    if( head->next != NULL )
    {
        data = head->next->data;  
        return (void *)data;
    }
    else
    {
        return NULL;
    }
}
