#include "hive.h"

volatile unsigned int
hiveAtomicSwap(volatile unsigned int *destination, unsigned int swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile u64 hiveAtomicSwapU64(volatile u64 *destination, u64 swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile void * 
hiveAtomicSwapPtr(volatile void ** destination, void * swapIn)
{
    return __sync_lock_test_and_set(destination, swapIn);
}

volatile unsigned int
hiveAtomicAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

volatile unsigned int
hiveAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

volatile u64
hiveAtomicFetchAddU64(volatile u64 *destination, u64 addVal)
{
    return __sync_fetch_and_add(destination, addVal);
}

volatile u64
hiveAtomicFetchSubU64(volatile u64 *destination, u64 subVal)
{
    return __sync_fetch_and_sub(destination, subVal);
}

volatile u64
hiveAtomicAddU64(volatile u64 *destination, u64 addVal)
{
    return __sync_add_and_fetch(destination, addVal);
}

volatile u64
hiveAtomicSubU64(volatile u64 *destination, u64 subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

volatile unsigned int
hiveAtomicSub(volatile unsigned int *destination, unsigned int subVal)
{
    return __sync_sub_and_fetch(destination, subVal);
}

volatile unsigned int
hiveAtomicCswap(volatile unsigned int *destination, unsigned int oldVal,
               unsigned int swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile u64 
hiveAtomicCswapU64(volatile u64 *destination, u64 oldVal,
               u64 swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile void * 
hiveAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn)
{
    return __sync_val_compare_and_swap(destination, oldVal, swapIn);

}

volatile bool
hiveAtomicSwapBool(volatile bool *destination, bool value)
{
    return __sync_lock_test_and_set(destination, value);
}

bool hiveLock( volatile unsigned int * lock)
{
    while(hiveAtomicCswap( lock, 0U, 1U ) == 1U);
    return true;
}

void hiveUnlock( volatile unsigned int * lock)
{
    //hiveAtomicSwap( lock, 0U );
    *lock=0U;
}

bool hiveTryLock( volatile unsigned int * lock)
{
    return (hiveAtomicCswap( lock, 0U, 1U ) == 0U);
}

volatile u64 hiveAtomicFetchAndU64(volatile u64 * destination, u64 addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}

volatile u64 hiveAtomicFetchOrU64(volatile u64 * destination, u64 addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

volatile unsigned int hiveAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_or(destination, addVal);
}

volatile unsigned int hiveAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal)
{
    return __sync_fetch_and_and(destination, addVal);
}