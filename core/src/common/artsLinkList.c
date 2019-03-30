//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include "artsLinkList.h"
#include "arts.h"
#include "artsAtomics.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

struct artsLinkListItem
{
    struct artsLinkListItem * next;
};

struct artsLinkList
{
    struct artsLinkListItem * headPtr;
    struct artsLinkListItem * tailPtr;
    volatile unsigned int lock;
};

void artsLinkListNew(struct artsLinkList * list)
{
    list->headPtr = list->tailPtr = NULL;
}

void artsLinkListDelete(void * linkList)
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

struct artsLinkList * artsLinkListGroupNew(unsigned int listSize)
{
    struct artsLinkList * linkList = (struct artsLinkList *) artsCalloc(sizeof(struct artsLinkList) * listSize);
    for (int i=0; i<listSize; i++)
    {
        artsLinkListNew(&linkList[i]);
    }
    return linkList;
}

void * artsLinkListNewItem(unsigned int size)
{
    struct artsLinkListItem * newItem = artsCalloc(sizeof(struct artsLinkListItem) + size);
    newItem->next=NULL;
    if(size)
    {
        return (void*)(newItem+1);
    }
    return NULL;
}

void artsLinkListDeleteItem(void * toDelete)
{
    struct artsLinkListItem * item = ((struct artsLinkListItem *) toDelete) - 1;
    artsFree(item);
}

inline struct artsLinkList * artsLinkListGet(struct artsLinkList *linkList, unsigned int position)
{
    return (struct artsLinkList *)(linkList + position);
}

void artsLinkListPushBack(struct artsLinkList * list, void * item)
{
    struct artsLinkListItem * newItem = item;
    newItem-=1;
    artsLock(&list->lock);
    if(list->headPtr == NULL)
    {
        list->headPtr = list->tailPtr = newItem;
    }
    else
    {
        list->tailPtr->next = newItem;
        list->tailPtr = newItem;
    }
    artsUnlock(&list->lock);
}

void * artsLinkListPopFront( struct artsLinkList * list, void ** freePos )
{
    void * data = NULL;
    *freePos = NULL;
    artsLock(&list->lock);
    if(list->headPtr)
    {
        data = (void*)(list->headPtr+1);
        list->headPtr=list->headPtr->next;
//        if(!list->headPtr)
//            list->tailPtr = NULL;
    }
    artsUnlock(&list->lock);
    return data;
}


