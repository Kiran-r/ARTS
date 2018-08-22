#ifndef ARTSARRAYDB_H
#define ARTSARRAYDB_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

typedef struct  artsArrayDb
{
    unsigned int elementSize;
    unsigned int elementsPerBlock;
    unsigned int numBlocks;
    char head[];
} artsArrayDb_t;

unsigned int artsGetSizeArrayDb(artsArrayDb_t * array);
artsGuid_t artsNewArrayDb(artsArrayDb_t **addr, unsigned int elementSize, unsigned int numElements);
artsArrayDb_t * artsNewArrayDbWithGuid(artsGuid_t guid, unsigned int elementSize, unsigned int numElements);
unsigned int getOffsetFromIndex(artsArrayDb_t * array, unsigned int index);
unsigned int getRankFromIndex(artsArrayDb_t * array, unsigned int index);
artsGuid_t getArrayDbGuid(artsArrayDb_t * array);
void artsGetFromArrayDb(artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);
void artsPutInArrayDb(void * ptr, artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);
void artsForEachInArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, u32 paramc, u64 * paramv);
artsGuid_t artsGatherArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u64 depc);

void artsAtomicAddInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot);
void internalAtomicAddInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsAtomicCompareAndSwapInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot);
void internalAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsForEachInArrayDbAtData(artsArrayDb_t * array, unsigned int stride, artsEdt_t funcPtr, u32 paramc, u64 * paramv);
#ifdef __cplusplus
}
#endif
#endif /* ARTSARRAYDB_H */

