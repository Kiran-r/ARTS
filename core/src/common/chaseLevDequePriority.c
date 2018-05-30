#include "hiveDeque.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"

struct circularArray
{
    struct circularArray * next;
    unsigned int size;
    void ** segment;
}__attribute__ ((aligned(64)));

struct hiveDeque
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
    struct hiveDeque * volatile left;
    struct hiveDeque * volatile right;
}__attribute__ ((aligned(64)));

static inline struct circularArray * 
newCircularArray(unsigned int size)
{
    struct circularArray * array 
        = hiveCalloc( sizeof(struct circularArray) + sizeof(void*) * size); 
//    memset(array,0,sizeof(struct circularArray) + sizeof(void*) * size);
    array->size = size;
    array->segment = (void**)(array+1);
    array->next = NULL;
    return array;
}


bool
hiveDequeEmpty(struct hiveDeque * deque)
{
    //don't really know what this is for
    //return (deque->bottom == deque->top);
    return false;
}

void
hiveDequeClear(struct hiveDeque *deque)
{
    deque->top = deque->bottom;
}

unsigned int 
hiveDequeSize(struct hiveDeque *deque)
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
hiveDequeNewInit(struct hiveDeque * deque, unsigned int size)
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

struct hiveDeque * 
hiveDequeNew(unsigned int size)
{
    struct hiveDeque * deque = hiveCalloc(sizeof(struct hiveDeque));
    hiveDequeNewInit(deque, size);
    return deque;
}

void
hiveDequeDelete(struct hiveDeque *deque)
{
    struct circularArray * trail, * current = deque->head;
    while(current)
    {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
//    free(deque);
}
struct hiveDeque * findTheDeque( struct hiveDeque * deque, unsigned int priority)
{
    struct hiveDeque * next = deque;
    
    while(next->priority != priority)
    {
        if(priority < next->priority)
        {
            unsigned int old;
            if(next->leftLock == 0)
            {
                old = hiveAtomicCswap(&next->leftLock, 0U, 1U);
                if(old == 0U)
                {
                    struct hiveDeque * nextDeque = hiveDequeNew(8);
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
                old = hiveAtomicCswap(&next->rightLock, 0U, 1U);
                if(old == 0U)
                {
                    struct hiveDeque * nextDeque = hiveDequeNew(8);
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
hiveDequePushFront(struct hiveDeque *deque, void *item, unsigned int priority)
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
hiveDequePopFront(struct hiveDeque *deque)
{
    void * o = NULL;
    
    if(deque->left)
        o = hiveDequePopFront(deque->left);
    
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
                if(hiveAtomicCswapU64(&deque->top, t, t+1) != t)
                    o = NULL;
                deque->bottom = t+1;
            }
        }

        if(!o && deque->right)
            o = hiveDequePopFront(deque->right);
    }
    return o;
}

void *
hiveDequePopBack(struct hiveDeque *deque)
{
    void * o = NULL;
   
    if(deque->left)
        o = hiveDequePopBack(deque->left);
    
    if(!o)
    {
        u64 t = deque->top;
        HW_MEMORY_FENCE();
        u64 b = deque->bottom;
        if(t < b)
        {
            o = getCircularArray(deque->activeArray, t);
            u64 temp = hiveAtomicCswapU64(&deque->top, t, t+1);
            if(temp!=t)
            {
                o=NULL;
            }
        }
        if(!o && deque->right)
            o = hiveDequePopBack(deque->right);
    }
    return o;
}

struct hiveDeque *
hiveDequeListNew(unsigned int listSize, unsigned int dequeSize)
{
    struct hiveDeque *dequeList =
        (struct hiveDeque *) hiveCalloc( listSize * sizeof (struct hiveDeque) );
    int i = 0;
    for (i = 0; i < listSize; i++)
        hiveDequeNewInit(&dequeList[i]  , dequeSize);

    return dequeList;
}

struct hiveDeque *
hiveDequeListGetDeque(struct hiveDeque *dequeList, unsigned int position)
{
    return dequeList+position;
}

void 
hiveDequeListDelete(void *dequeList)
{
//    hiveDeque * ptr = (hiveDeque*) dequeList;
//    for (i = 0; i < listSize; i++)
//        hiveDequeDelete( dequeList+i  , dequeSize);
//    hiveFree(dequeList);
}
