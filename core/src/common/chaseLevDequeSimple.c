#include <string.h>

#include "artsDeque.h"
#include "artsAtomics.h"

struct circularArray
{
    struct circularArray * next;
    unsigned int size;
    void ** segment;
}__attribute__ ((aligned(64)));

struct artsDeque
{
    volatile uint64_t  top;
    char pad1[56];
    volatile uint64_t bottom;
    char pad2[56];
    struct circularArray * volatile activeArray;
    char pad3[56];
    struct circularArray * head;
    volatile unsigned int push;
    volatile unsigned int pop;
    volatile unsigned int steal;
}__attribute__ ((aligned(64)));

static inline struct circularArray * 
newCircularArray(unsigned int size)
{
    struct circularArray * array 
        = artsCalloc( sizeof(struct circularArray) + sizeof(void*) * size); 
//    memset(array,0,sizeof(struct circularArray) + sizeof(void*) * size);
    array->size = size;
    array->segment = (void**)(array+1);
    array->next = NULL;
    return array;
}


bool
artsDequeEmpty(struct artsDeque * deque)
{
    //don't really know what this is for
    //return (deque->bottom == deque->top);
    return false;
}

void
artsDequeClear(struct artsDeque *deque)
{
    deque->top = deque->bottom;
}

unsigned int 
artsDequeSize(struct artsDeque *deque)
{
    return deque->bottom - deque->top;
}

static inline void * 
getCircularArray(struct circularArray * array, uint64_t i)
{
    return array->segment[i%array->size];
}

__thread void * stealArray[STEALSIZE];

static inline void 
getMultipleCircularArray(struct circularArray * array, uint64_t i)
{
    if(i%array->size + STEALSIZE < array->size)
        memcpy(stealArray, &array->segment[i%array->size],  sizeof(void*) * STEALSIZE);
    else
        for(unsigned int j=0; j<STEALSIZE; j++)
            stealArray[j] = array->segment[(i+j)%array->size];
}

static inline void 
putCircularArray(struct circularArray * array, uint64_t i, void * object)
{
    array->segment[i%array->size] = object;
}

static inline struct circularArray *
growCircularArray(struct circularArray * array, uint64_t b, uint64_t t)
{
    struct circularArray * a = newCircularArray(array->size*2);
    array->next = a;
    uint64_t i;
    for(i=t; i<b; i++)
        putCircularArray(a, i, getCircularArray(array, i));
    return a;
}

static inline void
artsDequeNewInit(struct artsDeque * deque, unsigned int size)
{
    deque->top = 1;
    deque->bottom = 1;
    deque->activeArray = newCircularArray(size);
    deque->head = deque->activeArray;
    deque->push = 0;
    deque->pop = 0;
    deque->steal = 0;
}

struct artsDeque * 
artsDequeNew(unsigned int size)
{
    struct artsDeque * deque = artsCalloc(sizeof(struct artsDeque));
    artsDequeNewInit(deque, size);
    return deque;
}

void
artsDequeDelete(struct artsDeque *deque)
{
    struct circularArray * trail, * current = deque->head;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
//    free(deque);
}

bool 
artsDequePushFront(struct artsDeque *deque, void *item, unsigned int priority)
{
    struct circularArray * a = deque->activeArray;
    uint64_t b = deque->bottom;
    uint64_t t = deque->top;
    if(b >= a->size - 1 + t)
    {
        a = growCircularArray(a, b, t);
        deque->activeArray = a;
    }
    putCircularArray(a, b, item);
    HW_MEMORY_FENCE();
    deque->bottom=b+1;
    return true;
}

void *
artsDequePopFront(struct artsDeque *deque)
{
    uint64_t b = --deque->bottom;
    HW_MEMORY_FENCE();
    uint64_t t = deque->top;
    if(t > b)
    {
        deque->bottom = t;        
        return NULL;
    }
    void * o = getCircularArray(deque->activeArray, b);
    //Success
    if(b > t)
    {
        return o;
    }
    if(artsAtomicCswapU64(&deque->top, t, t+1) != t)
        o = NULL;
    deque->bottom = t+1;
    return o;
}

void *
artsDequePopBack(struct artsDeque *deque)
{
    uint64_t t = deque->top;
    HW_MEMORY_FENCE();
    uint64_t b = deque->bottom;
    if(t < b)
    {
        void * o = getCircularArray(deque->activeArray, t);
        uint64_t temp = artsAtomicCswapU64(&deque->top, t, t+1);
        if(temp==t)
        {        
            return o;
        }
    }
    return NULL;
}

struct artsDeque *
artsDequeListNew(unsigned int listSize, unsigned int dequeSize)
{
    struct artsDeque *dequeList =
        (struct artsDeque *) artsCalloc( listSize * sizeof (struct artsDeque) );
    int i = 0;
    for (i = 0; i < listSize; i++)
        artsDequeNewInit(&dequeList[i]  , dequeSize);

    return dequeList;
}

struct artsDeque *
artsDequeListGetDeque(struct artsDeque *dequeList, unsigned int position)
{
    return dequeList+position;
}

void 
artsDequeListDelete(void *dequeList)
{
//    artsDeque * ptr = (artsDeque*) dequeList;
//    for (i = 0; i < listSize; i++)
//        artsDequeDelete( dequeList+i  , dequeSize);
//    artsFree(dequeList);
}
