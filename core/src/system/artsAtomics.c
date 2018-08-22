#include "arts.h"

volatile unsigned int
artsAtomicSwap(volatile unsigned int *destination, unsigned int swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile uint64_t artsAtomicSwapU64(volatile uint64_t *destination, uint64_t swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile void * 
artsAtomicSwapPtr(volatile void ** destination, void * swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile unsigned int
artsAtomicAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

volatile unsigned int
artsAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

volatile uint64_t
artsAtomicFetchAddU64(volatile uint64_t *destination, uint64_t addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

volatile uint64_t
artsAtomicFetchSubU64(volatile uint64_t *destination, uint64_t subVal)
{
    return __sync_fetch_and_sub(destination, subVal);
}

volatile uint64_t
artsAtomicAddU64(volatile uint64_t *destination, uint64_t addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

volatile uint64_t
artsAtomicSubU64(volatile uint64_t *destination, uint64_t subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

volatile unsigned int
artsAtomicSub(volatile unsigned int *destination, unsigned int subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

volatile unsigned int
artsAtomicCswap(volatile unsigned int *destination, unsigned int oldVal,
               unsigned int swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile uint64_t 
artsAtomicCswapU64(volatile uint64_t *destination, uint64_t oldVal,
               uint64_t swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile void * 
artsAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile bool
artsAtomicSwapBool(volatile bool *destination, bool value)
{
    return __sync_lock_test_and_set(destination, value);
}

bool artsLock( volatile unsigned int * lock)
{
    while(artsAtomicCswap( lock, 0U, 1U ) == 1U);
    return true;
}

void artsUnlock( volatile unsigned int * lock)
{
    //artsAtomicSwap( lock, 0U );
    *lock=0U;
}

bool artsTryLock( volatile unsigned int * lock)
{
    return (artsAtomicCswap( lock, 0U, 1U ) == 0U);
}

volatile uint64_t artsAtomicFetchAndU64(volatile uint64_t * destination, uint64_t addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}

volatile uint64_t artsAtomicFetchOrU64(volatile uint64_t * destination, uint64_t addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

volatile unsigned int artsAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

volatile unsigned int artsAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}