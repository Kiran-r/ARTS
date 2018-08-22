#include "artsDeque.h"
#include "artsMalloc.h"
#include "artsAtomics.h"

struct circularArray
{
    struct circularArray * next;
    unsigned int size;
    void ** segment;
}__attribute__ ((aligned(64)));

struct artsDeque
{
    volatile u64  top;
    char pad1[56];
    volatile u64 bottom;
    char pad2[56];
    struct circularArray * volatile activeArray;
    char pad3[56];
    volatile unsigned int leftLock;
    char pad4[60];
    volatile unsigned int rightLock;
    char pad5[60];
    struct circularArray * head;
    volatile unsigned int push;
    volatile unsigned int pop;
    volatile unsigned int steal;
    unsigned int priority;
    struct artsDeque * volatile left;
    struct artsDeque * volatile right;
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
getCircularArray(struct circularArray * array, u64 i)
{
    return array->segment[i%array->size];
}

static inline void 
putCircularArray(struct circularArray * array, u64 i, void * object)
{
    array->segment[i%array->size] = object;
}

static inline struct circularArray *
growCircularArray(struct circularArray * array, u64 b, u64 t)
{
    struct circularArray * a = newCircularArray(array->size*2);
    array->next = a;
    u64 i;
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
    deque->priority = 0;
    deque->left = deque->right=NULL;
    deque->leftLock = deque->rightLock = 0;
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
struct artsDeque * findTheDeque( struct artsDeque * deque, unsigned int priority)
{
    struct artsDeque * next = deque;
    
    while(next->priority != priority)
    {
        if(priority < next->priority)
        {
            unsigned int old;
            if(next->leftLock == 0)
            {
                old = artsAtomicCswap(&next->leftLock, 0U, 1U);
                if(old == 0U)
                {
                    struct artsDeque * nextDeque = artsDequeNew(8);
                    nextDeque->priority = priority;
                    next->left = nextDeque;
                    return next->left;
                }
            }
            while(next->left==NULL){}
            next = next->left;
        }
        else
        {
            unsigned int old;
            if(next->rightLock == 0)
            {
                old = artsAtomicCswap(&next->rightLock, 0U, 1U);
                if(old == 0U)
                {
                    struct artsDeque * nextDeque = artsDequeNew(8);
                    nextDeque->priority = priority;
                    next->right = nextDeque;
                    return next->right;
                }
            }
            while(next->right==NULL){}
            next = next->right;
            
        }
    }

    return next;
}

bool 
artsDequePushFront(struct artsDeque *deque, void *item, unsigned int priority)
{
    deque = findTheDeque(deque, priority);
    struct circularArray * a = deque->activeArray;
    u64 b = deque->bottom;
    u64 t = deque->top;
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
    void * o = NULL;
    
    if(deque->left)
        o = artsDequePopFront(deque->left);
    
    if(!o)
    {

        u64 b = --deque->bottom;
        HW_MEMORY_FENCE();
        u64 t = deque->top;
        
        o = getCircularArray(deque->activeArray, b);
        if(t > b)
        {
            deque->bottom = t;

            o = NULL;
        }
        else
        {
            //Success
            getCircularArray(deque->activeArray, b);
            if(b <= t)
            {
                if(artsAtomicCswapU64(&deque->top, t, t+1) != t)
                    o = NULL;
                deque->bottom = t+1;
            }
        }

        if(!o && deque->right)
            o = artsDequePopFront(deque->right);
    }
    return o;
}

void *
artsDequePopBack(struct artsDeque *deque)
{
    void * o = NULL;
   
    if(deque->left)
        o = artsDequePopBack(deque->left);
    
    if(!o)
    {
        u64 t = deque->top;
        HW_MEMORY_FENCE();
        u64 b = deque->bottom;
        if(t < b)
        {
            o = getCircularArray(deque->activeArray, t);
            u64 temp = artsAtomicCswapU64(&deque->top, t, t+1);
            if(temp!=t)
            {
                o=NULL;
            }
        }
        if(!o && deque->right)
            o = artsDequePopBack(deque->right);
    }
    return o;
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
