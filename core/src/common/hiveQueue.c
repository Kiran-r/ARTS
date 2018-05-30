#include "hiveQueue.h"
#include "hiveMalloc.h"
#include "hive.h"
#define ALIGNMENT 8
// inline int is_empty(uint64_t v) __attribute__ ((pure));
// inline uint64_t node_index(uint64_t i) __attribute__ ((pure));
// inline uint64_t set_unsafe(uint64_t i) __attribute__ ((pure));
// inline uint64_t node_unsafe(uint64_t i) __attribute__ ((pure));
// inline uint64_t tail_index(uint64_t t) __attribute__ ((pure));
// inline int crq_is_closed(uint64_t t) __attribute__ ((pure));

//RingQueue *head;
//RingQueue *tail;
//__thread RingQueue *nrq;

void * hiveMallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = hiveMalloc(size + align);
    memset(ptr, 0, align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 'a';
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void * hiveCallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = hiveCalloc(size + align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 1;
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void hiveFreeAlign(void * ptr)
{
    char * trail = (char*)ptr - 1;
    while(!(*trail))
        trail--;
    hiveFree(trail);
}

// inline void init_ring(RingQueue *r)
void init_ring(RingQueue *r)
{
    for(int i = 0; i < RING_SIZE; i++)
    {
        r->array[i].val = -1;
        r->array[i].idx = i;
    }
    r->head = r->tail = 0;
    r->next = NULL;
}

// inline int is_empty(uint64_t v)
int is_empty(uint64_t v)
{
    return (v == (uint64_t)-1);
}

// inline uint64_t node_index(uint64_t i)
uint64_t node_index(uint64_t i)
{
    return (i & ~(1ull << 63));
}

// inline uint64_t set_unsafe(uint64_t i)
uint64_t set_unsafe(uint64_t i)
{
    return (i | (1ull << 63));
}

// inline uint64_t node_unsafe(uint64_t i)
uint64_t node_unsafe(uint64_t i)
{
    return (i & (1ull << 63));
}

// inline uint64_t tail_index(uint64_t t)
uint64_t tail_index(uint64_t t)
{
    return (t & ~(1ull << 63));
}

// inline int crq_is_closed(uint64_t t)
int crq_is_closed(uint64_t t)
{
    return (t & (1ull << 63)) != 0;
}

hiveQueue * hiveNewQueue()
{
    hiveQueue * queue = hiveCallocAlign(sizeof(hiveQueue), 128);
    RingQueue *rq = hiveCallocAlign(sizeof(RingQueue), 128);
    init_ring(rq);
    queue->head = queue->tail = rq;
    return queue;
}

// inline void fixState(RingQueue *rq)
void fixState(RingQueue *rq)
{
    uint64_t t, h, n;
    while (1)
    {
        uint64_t t = FAA64(&rq->tail, 0);
        uint64_t h = FAA64(&rq->head, 0);

        if (unlikely(rq->tail != t))
            continue;

        if (h > t)
        {
            if (CAS64(&rq->tail, t, h)) break;
            continue;
        }
        break;
    }
}

// inline int close_crq(RingQueue *rq, const uint64_t t, const int tries)
int close_crq(RingQueue *rq, const uint64_t t, const int tries)
{
    if (tries < 10)
        return CAS64(&rq->tail, t + 1, (t + 1)|(1ull<<63));
    else
        return BIT_TEST_AND_SET(&rq->tail, 63);
}

void enqueue(Object arg, hiveQueue * queue)
{
    RingQueue * nrq;
    int try_close = 0;
    while (1)
    {
        RingQueue *rq = queue->tail;
        RingQueue *next = rq->next;

        if (unlikely(next != NULL))
        {
            CASPTR(&queue->tail, rq, next);
            continue;
        }

        uint64_t t = FAA64(&rq->tail, 1);

        if (crq_is_closed(t))
        {
alloc:
//            PRINTF("Allocing!\n");
            nrq = hiveMallocAlign(sizeof(RingQueue), 128);
            init_ring(nrq);

            // Solo enqueue
            nrq->tail = 1;
            nrq->array[0].val = arg;
            nrq->array[0].idx = 0;

            if (CASPTR(&rq->next, NULL, nrq))
            {
                CASPTR(&queue->tail, rq, nrq);
                nrq = NULL;
                return;
            }
            else
            {
                hiveFreeAlign(nrq);
            }
            continue;
        }

        RingNode* cell = &rq->array[t & (RING_SIZE-1)];
        StorePrefetch(cell);

        uint64_t idx = cell->idx;
        uint64_t val = cell->val;

        if (likely(is_empty(val)))
        {
            if (likely(node_index(idx) <= t))
            {
                if ((likely(!node_unsafe(idx)) || rq->head < t) && CAS2((uint64_t*)cell, -1, idx, arg, t))
                {
                    return;
                }
            }
        }

        uint64_t h = rq->head;
        if (unlikely(t >= RING_SIZE + h) && close_crq(rq, t, ++try_close))
        {
            goto alloc;
        }
    }
}

Object dequeue(hiveQueue * queue)
{
    while (1)
    {
        RingQueue *rq = queue->head;
        RingQueue *next;
        uint64_t h = FAA64(&rq->head, 1);
        RingNode* cell = &rq->array[h & (RING_SIZE-1)];
        StorePrefetch(cell);

        uint64_t tt;
        int r = 0;

        while (1)
        {
            uint64_t cell_idx = cell->idx;
            uint64_t unsafe = node_unsafe(cell_idx);
            uint64_t idx = node_index(cell_idx);
            uint64_t val = cell->val;

            if (unlikely(idx > h)) break;

            if (likely(!is_empty(val)))
            {
                if (likely(idx == h))
                {
                    if (CAS2((uint64_t*)cell, val, cell_idx, -1, unsafe | h + RING_SIZE))
                        return val;
                }
                else
                {
                    if (CAS2((uint64_t*)cell, val, cell_idx, val, set_unsafe(idx)))
                    {
                        break;
                    }
                }
            }
            else
            {
                if( (r & ((1ull << 10) - 1)) == 0)
                    tt = rq->tail;

                // Optimization: try to bail quickly if queue is closed.
                int crq_closed = crq_is_closed(tt);
                uint64_t t = tail_index(tt);

                if (unlikely(unsafe)) // Nothing to do, move along
                {
                    if (CAS2((uint64_t*)cell, val, cell_idx, val, unsafe | h + RING_SIZE))
                        break;
                }
                else if (t <= h + 1 || r > 200000 || crq_closed)
                {
                    if (CAS2((uint64_t*)cell, val, idx, val, h + RING_SIZE))
                        break;
                }
                else
                {
                    ++r;
                }
            }
        }

        if (tail_index(rq->tail) <= h + 1)
        {
            fixState(rq);
            // try to return empty
            next = rq->next;
            if (next == NULL)
            {
                return 0;  // EMPTY
            }
            CASPTR(&queue->head, rq, next);
        }
    }
}
