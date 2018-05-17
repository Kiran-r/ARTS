#include "hive.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveLinkList.h"
#include <string.h>

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
struct hiveLinkListItem
{
    //char pad[48];
    struct hiveLinkListItem * volatile next;
    void * volatile  data;
    //volatile unsigned int readers;
    //char data[];
} __attribute__ ((aligned));

struct hiveLinkList
{
    struct hiveLinkListItem * volatile headPtr;
    struct hiveLinkListItem * volatile tailPtr;
    volatile unsigned int lock;
} __attribute__ ((aligned));


void
hiveLinkListNew(struct hiveLinkList *list)
{
    struct hiveLinkListItem * newItem = hiveMalloc( sizeof(struct hiveLinkListItem) );
    newItem->next=NULL;
    list->headPtr = list->tailPtr = newItem;
}


struct hiveLinkList *
hiveLinkListGroupNew(unsigned int listSize)
{
    DPRINTF("%d\n", sizeof( struct hiveLinkListItem ));
    int i;
    struct hiveLinkList *linkList =
        (struct hiveLinkList *) hiveCalloc(  sizeof (struct hiveLinkList) * listSize );

    for (i = 0; i < listSize; i++)
        hiveLinkListNew( linkList+i);

    return linkList;
}


inline struct hiveLinkList *
hiveLinkListGet(struct hiveLinkList *linkList, unsigned int position)
{
    //PRINTF("%d\n",dequeList->size);
    return (struct hiveLinkList *)( linkList + position  );
}


void
hiveLinkListDelete(void *linkList)
{
    struct hiveLinkList * list = linkList;
    struct hiveLinkListItem * last;
    while(list->headPtr != NULL)
    {
        last = (struct hiveLinkListItem *)list->headPtr;
        list->headPtr = list->headPtr->next;
        hiveFree(last);
    }
    hiveFree(linkList);
}
volatile unsigned int push=0,pop=0;
//Only 1 Pusher
void * hiveLinkListNewItem(unsigned int size)
{
    struct hiveLinkListItem * newItem = hiveMalloc( sizeof(struct hiveLinkListItem) + size );
    newItem->next=NULL;
    newItem->data = newItem+1;

    return (void*)newItem->data;
}

void 
hiveLinkListPushBack(struct hiveLinkList *list, void *item, unsigned int size)
{
    //struct hiveLinkListItem * newItem = hiveMalloc( sizeof(struct hiveLinkListItem) );
    struct hiveLinkListItem * newItem = item;
    newItem-=1;
    //newItem->next=NULL;
    //newItem->readers=0;
    //unsigned int res = hiveAtomicAdd(&push, 1U);
    //DPRINTF("PUSH %p %p %d %d\n", newItem, list->headPtr, res, pop);

    //newItem->data = item;

    //memcpy(newItem->data, item, size );

    void * ptr;
    volatile struct hiveLinkListItem * volatile tail;
    volatile struct hiveLinkListItem * volatile next;


    while(1)
    {
        if(list->lock == 0U)
        {
            if(hiveAtomicCswap( &list->lock, 0U, 1U ) == 0U)
                break;
        }
    }
   

    list->tailPtr->next = newItem;
    list->tailPtr = newItem;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    list->lock = 0U;

    //tailPtr = list->tailPtr;
    //while(1)
    {
        /*if(list->headPtr==NULL)
        {
            
            ptr = hiveAtomicCswapPtr( &list->headPtr, NULL, newItem );
            if(ptr == NULL)
            {
                list->tailPtr = newItem;
                //list->headPtr = newItem;
                break;
            }
        }
        else
        {*/
            //tail = (void *)list->tailPtr;
            //next = tail->next;
            
            //if(tail != NULL)
            //{
                //hiveAtomicAdd(&tail->readers, 1U);
            
            /*if(list->tailPtr == tail)
            {
                if(next != NULL)
                    hiveAtomicCswapPtr( (volatile void** volatile  )&list->tailPtr, tail, next ); 
                else
                {
                    ptr = (void *)hiveAtomicCswapPtr( (volatile void** volatile  )&list->tailPtr->next, NULL, newItem ); 
               
                    if(ptr == NULL)
                    {
                        //hiveAtomicSub(&tail->readers, 1U);
                        break;
                    }
                }
            }*/
                //hiveAtomicSub(&tail->readers, 1U);
           // }
        //}
    }
}

//Only 1 popper
void * hiveLinkListPopFront( struct hiveLinkList * list, void ** freePos )
{
    volatile struct hiveLinkListItem * volatile head = list->headPtr;
    volatile struct hiveLinkListItem * volatile tail = list->tailPtr;
    volatile void * volatile data;

    void * ptr;

    //if(head != NULL)
    //{
    if(head != tail)
    {
        data = head->next->data;
        list->headPtr=head->next;

        *freePos= (void *)head;
        //unsigned int res = hiveAtomicAdd(&pop, 1U);
        //DPRINTF("POP %p %d %d\n", head, push, res);
        //hiveFree(head);
        return (void *)data;
        //
        //return NULL;
    }
    else
        return NULL;
}

void hiveLinkListDeleteItem( void * item  )
{
   /*char * addr = item;
   addr -= sizeof(struct hiveLinkListItem);

   hiveFree(addr);*/
}
