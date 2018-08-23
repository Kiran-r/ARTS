#ifndef ARTSARRAYDB_H
#define ARTSARRAYDB_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

unsigned int getOffsetFromIndex(artsArrayDb_t * array, unsigned int index);
unsigned int getRankFromIndex(artsArrayDb_t * array, unsigned int index);
artsGuid_t getArrayDbGuid(artsArrayDb_t * array);
void internalAtomicAddInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void internalAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);

#ifdef __cplusplus
}
#endif
#endif /* ARTSARRAYDB_H */

