#ifndef ARTSATOMICS_H
#define ARTSATOMICS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#define HW_MEMORY_FENCE() __sync_synchronize() 
#define COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT() __asm__ volatile("": : :"memory")

volatile unsigned int artsAtomicSwap(volatile unsigned int *destination, unsigned int swapIn);
volatile uint64_t artsAtomicSwapU64(volatile uint64_t *destination, uint64_t swapIn);
volatile void * artsAtomicSwapPtr(volatile void *destination, void * swapIn);
volatile unsigned int artsAtomicSub(volatile unsigned int *destination, unsigned int subVal);
volatile unsigned int artsAtomicAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int artsAtomicCswap(volatile unsigned int *destination, unsigned int oldVal, unsigned int swapIn);
volatile uint64_t artsAtomicCswapU64(volatile uint64_t *destination, uint64_t oldVal, uint64_t swapIn);
volatile void * artsAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn);
volatile bool artsAtomicSwapBool(volatile bool *destination, bool value);
volatile uint64_t artsAtomicFetchAddU64(volatile uint64_t *destination, uint64_t addVal);
volatile uint64_t artsAtomicFetchSubU64(volatile uint64_t *destination, uint64_t subVal);
volatile uint64_t artsAtomicAddU64(volatile uint64_t *destination, uint64_t addVal);
volatile uint64_t artsAtomicSubU64(volatile uint64_t *destination, uint64_t subVal);
bool artsLock( volatile unsigned int * lock);
void artsUnlock( volatile unsigned int * lock);
bool artsTryLock( volatile unsigned int * lock);
volatile uint64_t artsAtomicFetchAndU64(volatile uint64_t * destination, uint64_t addVal);
volatile uint64_t artsAtomicFetchOrU64(volatile uint64_t * destination, uint64_t addVal);
volatile unsigned int artsAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal);
volatile unsigned int artsAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal);
#ifdef __cplusplus
}
#endif

#endif
