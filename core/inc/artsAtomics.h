#ifndef ARTSATOMICS_H
#define ARTSATOMICS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#define HW_MEMORY_FENCE() __sync_synchronize() 
#define COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT() __asm__ volatile("": : :"memory")

volatile unsigned int artsAtomicSwap(volatile unsigned int *destination, unsigned int swapIn);
volatile u64 artsAtomicSwapU64(volatile u64 *destination, u64 swapIn);
volatile void * artsAtomicSwapPtr(volatile void *destination, void * swapIn);
volatile unsigned int artsAtomicSub(volatile unsigned int *destination, unsigned int subVal);
volatile unsigned int artsAtomicAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicCswap(volatile unsigned int *destination, unsigned int oldVal, unsigned int swapIn);
volatile u64 artsAtomicCswapU64(volatile u64 *destination, u64 oldVal, u64 swapIn);
volatile void * artsAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn);
volatile bool artsAtomicSwapBool(volatile bool *destination, bool value);
volatile u64 artsAtomicFetchAddU64(volatile u64 *destination, u64 addVal);
volatile u64 artsAtomicFetchSubU64(volatile u64 *destination, u64 subVal);
volatile u64 artsAtomicAddU64(volatile u64 *destination, u64 addVal);
volatile u64 artsAtomicSubU64(volatile u64 *destination, u64 subVal);
bool artsLock( volatile unsigned int * lock);
void artsUnlock( volatile unsigned int * lock);
bool artsTryLock( volatile unsigned int * lock);
volatile u64 artsAtomicFetchAndU64(volatile u64 * destination, u64 addVal);
volatile u64 artsAtomicFetchOrU64(volatile u64 * destination, u64 addVal);
volatile unsigned int artsAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal);
#ifdef __cplusplus
}
#endif

#endif
