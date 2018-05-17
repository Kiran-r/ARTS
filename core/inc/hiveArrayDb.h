#ifndef HIVEARRAYDB_H
#define HIVEARRAYDB_H

#include "hive.h"

typedef struct  hiveArrayDb
{
    unsigned int elementSize;
    unsigned int elementsPerBlock;
    unsigned int numBlocks;
    char head[];
} hiveArrayDb_t;

unsigned int hiveGetSizeArrayDb(hiveArrayDb_t * array);
hiveGuid_t hiveNewArrayDb(hiveArrayDb_t **addr, unsigned int elementSize, unsigned int numElements);
hiveArrayDb_t * hiveNewArrayDbWithGuid(hiveGuid_t guid, unsigned int elementSize, unsigned int numElements);
unsigned int getOffsetFromIndex(hiveArrayDb_t * array, unsigned int index);
unsigned int getRankFromIndex(hiveArrayDb_t * array, unsigned int index);
hiveGuid_t getArrayDbGuid(hiveArrayDb_t * array);
void hiveGetFromArrayDb(hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index);
void hivePutInArrayDb(void * ptr, hiveGuid_t edtGuid, unsigned int slot, hiveArrayDb_t * array, unsigned int index);
void hiveForEachInArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, u32 paramc, u64 * paramv);
hiveGuid_t hiveGatherArrayDb(hiveArrayDb_t * array, hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u64 depc);

void hiveAtomicAddInArrayDb(hiveArrayDb_t * array, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot);
void internalAtomicAddInArrayDb(hiveGuid_t dbGuid, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveAtomicCompareAndSwapInArrayDb(hiveArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot);
void internalAtomicCompareAndSwapInArrayDb(hiveGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveForEachInArrayDbAtData(hiveArrayDb_t * array, unsigned int stride, hiveEdt_t funcPtr, u32 paramc, u64 * paramv);
#endif /* HIVEARRAYDB_H */

