#ifndef ARTSEDTFUNCTIONS_H
#define ARTSEDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

bool artsEdtCreateInternal(artsGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, artsGuid_t eventGuid, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool useEpoch, artsGuid_t epochGuid, bool hasDepv);
void artsEdtDelete(struct artsEdt * edt);
void internalSignalEdt(artsGuid_t edtPacket, uint32_t slot, artsGuid_t dataGuid, artsType_t mode, void * ptr, unsigned int size);

typedef struct 
{
    artsGuid_t currentEdtGuid;
    struct artsEdt * currentEdt;
    void * epochList;
} threadLocal_t;

void artsSetThreadLocalEdtInfo(struct artsEdt * edt);
void artsUnsetThreadLocalEdtInfo();
void artsSaveThreadLocal(threadLocal_t * tl);
void artsRestoreThreadLocal(threadLocal_t * tl);


bool artsSetCurrentEpochGuid(artsGuid_t epochGuid);
artsGuid_t * artsCheckEpochIsRoot(artsGuid_t toCheck);
void artsIncrementFinishedEpochList();

typedef struct {
    void * buffer;
    uint32_t * sizeToWrite;
    unsigned int size;
    artsGuid_t epochGuid;
    volatile unsigned int uses;
} artsBuffer_t;

#ifdef __cplusplus
}
#endif

#endif
