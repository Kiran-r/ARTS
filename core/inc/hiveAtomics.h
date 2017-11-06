#ifndef HIVEATOMICS_H
#define HIVEATOMICS_H

#include "hive.h"
#define HW_MEMORY_FENCE() __sync_synchronize() 
#define COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT() __asm__ volatile("": : :"memory")

volatile unsigned int hiveAtomicSwap(volatile unsigned int *destination, unsigned int swapIn);
volatile u64 hiveAtomicSwapU64(volatile u64 *destination, u64 swapIn);
volatile void * hiveAtomicSwapPtr(volatile void *destination, void * swapIn);
volatile unsigned int hiveAtomicSub(volatile unsigned int *destination, unsigned int subVal);
volatile unsigned int hiveAtomicAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int hiveAtomicFetchAdd(volatile unsigned int *destination, unsigned int addVal);
volatile unsigned int hiveAtomicCswap(volatile unsigned int *destination, unsigned int oldVal, unsigned int swapIn);
volatile u64 hiveAtomicCswapU64(volatile u64 *destination, u64 oldVal, u64 swapIn);
volatile void * hiveAtomicCswapPtr(volatile void **destination, void * oldVal, void * swapIn);
volatile bool hiveAtomicSwapBool(volatile bool *destination, bool value);
volatile u64 hiveAtomicFetchAddU64(volatile u64 *destination, u64 addVal);
volatile u64 hiveAtomicFetchSubU64(volatile u64 *destination, u64 subVal);
volatile u64 hiveAtomicAddU64(volatile u64 *destination, u64 addVal);
volatile u64 hiveAtomicSubU64(volatile u64 *destination, u64 subVal);
bool hiveLock( volatile unsigned int * lock);
void hiveUnlock( volatile unsigned int * lock);
bool hiveTryLock( volatile unsigned int * lock);
volatile u64 hiveAtomicFetchAndU64(volatile u64 * destination, u64 addVal);
volatile u64 hiveAtomicFetchOrU64(volatile u64 * destination, u64 addVal);
volatile unsigned int hiveAtomicFetchOr(volatile unsigned int * destination, unsigned int addVal);
volatile unsigned int hiveAtomicFetchAnd(volatile unsigned int * destination, unsigned int addVal);
#endif
