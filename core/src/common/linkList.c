#include "arts.h"
#include "artsMalloc.h"
#include "artsAtomics.h"
#include "artsLinkList.h"
#include <string.h>

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
struct artsLinkListItem
{
    //char pad[48];
    struct artsLinkListItem * volatile next;
    void * volatile  data;
    //volatile unsigned int readers;
    //char data[];
} __attribute__ ((aligned));

struct artsLinkList
{
    struct artsLinkListItem * volatile headPtr;
    struct artsLinkListItem * volatile tailPtr;
    volatile unsigned int lock;
} __attribute__ ((aligned));


void
artsLinkListNew(struct artsLinkList *list)
{
    struct artsLinkListItem * newItem = artsMalloc( sizeof(struct artsLinkListItem) );
    newItem->next=NULL;
    list->headPtr = list->tailPtr = newItem;
}


struct artsLinkList *
artsLinkListGroupNew(unsigned int listSize)
{
    DPRINTF("%d\n", sizeof( struct artsLinkListItem ));
    int i;
    struct artsLinkList *linkList =
        (struct artsLinkList *) artsCalloc(  sizeof (struct artsLinkList) * listSize );

    for (i = 0; i < listSize; i++)
        artsLinkListNew( linkList+i);

    return linkList;
}


inline struct artsLinkList *
artsLinkListGet(struct artsLinkList *linkList, unsigned int position)
{
    //PRINTF("%d\n",dequeList->size);
    return (struct artsLinkList *)( linkList + position  );
}


void
artsLinkListDelete(void *linkList)
{
    struct artsLinkList * list = linkList;
    struct artsLinkListItem * last;
    while(list->headPtr != NULL)
    {
        last = (struct artsLinkListItem *)list->headPtr;
        list->headPtr = list->headPtr->next;
        artsFree(last);
    }
    artsFree(linkList);
}
volatile unsigned int push=0,pop=0;
//Only 1 Pusher
void * artsLinkListNewItem(unsigned int size)
{
    struct artsLinkListItem * newItem = artsMalloc( sizeof(struct artsLinkListItem) + size );
    newItem->next=NULL;
    newItem->data = newItem+1;

    return (void*)newItem->data;
}

void 
artsLinkListPushBack(struct artsLinkList *list, void *item, unsigned int size)
{
    //struct artsLinkListItem * newItem = artsMalloc( sizeof(struct artsLinkListItem) );
    struct artsLinkListItem * newItem = item;
    newItem-=1;
    //newItem->next=NULL;
    //newItem->readers=0;
    //unsigned int res = artsAtomicAdd(&push, 1U);
    //DPRINTF("PUSH %p %p %d %d\n", newItem, list->headPtr, res, pop);

    //newItem->data = item;

    //memcpy(newItem->data, item, size );

    void * ptr;
    volatile struct artsLinkListItem * volatile tail;
    volatile struct artsLinkListItem * volatile next;


    while(1)
    {
        if(list->lock == 0U)
        {
            if(artsAtomicCswap( &list->lock, 0U, 1U ) == 0U)
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
            
            ptr = artsAtomicCswapPtr( &list->headPtr, NULL, newItem );
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
                //artsAtomicAdd(&tail->readers, 1U);
            
            /*if(list->tailPtr == tail)
            {
                if(next != NULL)
                    artsAtomicCswapPtr( (volatile void** volatile  )&list->tailPtr, tail, next ); 
                else
                {
                    ptr = (void *)artsAtomicCswapPtr( (volatile void** volatile  )&list->tailPtr->next, NULL, newItem ); 
               
                    if(ptr == NULL)
                    {
                        //artsAtomicSub(&tail->readers, 1U);
                        break;
                    }
                }
            }*/
                //artsAtomicSub(&tail->readers, 1U);
           // }
        //}
    }
}

//Only 1 popper
void * artsLinkListPopFront( struct artsLinkList * list, void ** freePos )
{
    volatile struct artsLinkListItem * volatile head = list->headPtr;
    volatile struct artsLinkListItem * volatile tail = list->tailPtr;
    volatile void * volatile data;

    void * ptr;

    //if(head != NULL)
    //{
    if(head != tail)
    {
        data = head->next->data;
        list->headPtr=head->next;

        *freePos= (void *)head;
        //unsigned int res = artsAtomicAdd(&pop, 1U);
        //DPRINTF("POP %p %d %d\n", head, push, res);
        //artsFree(head);
        return (void *)data;
        //
        //return NULL;
    }
    else
        return NULL;
}

void artsLinkListDeleteItem( void * item  )
{
   /*char * addr = item;
   addr -= sizeof(struct artsLinkListItem);

   artsFree(addr);*/
}
