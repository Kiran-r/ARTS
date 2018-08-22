#ifndef ARTSEDTFUNCTIONS_H
#define ARTSEDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h>

#define artsEdtArgs(...) sizeof((u64[]){__VA_ARGS__})/sizeof(u64), (u64[]){__VA_ARGS__}
#define artsEdtEmptySignal(guid) artsSignalEdt(guid, NULL_GUID, -1);

bool artsEdtCreateInternal(artsGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, artsGuid_t eventGuid, artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, bool useEpoch, artsGuid_t epochGuid, bool hasDepv);

artsGuid_t artsEdtCreate(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc);
artsGuid_t artsEdtCreateWithGuid(artsEdt_t funcPtr, artsGuid_t guid, u32 paramc, u64 * paramv, u32 depc);
artsGuid_t artsEdtCreateWithEpoch(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, artsGuid_t epochGuid);

artsGuid_t artsEdtCreateDep(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
artsGuid_t artsEdtCreateWithGuidDep(artsEdt_t funcPtr, artsGuid_t guid, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
artsGuid_t artsEdtCreateWithEpochDep(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, artsGuid_t epochGuid, bool hasDepv);

void artsEdtDelete(struct artsEdt * edt);
void artsEdtDestroy(artsGuid_t guid);

void internalSignalEdt(artsGuid_t edtPacket, u32 slot, artsGuid_t dataGuid, artsType_t mode, void * ptr, unsigned int size);

void artsSignalEdt(artsGuid_t edtGuid, u32 slot, artsGuid_t dataGuid);
void artsSignalEdtValue(artsGuid_t edtGuid, u32 slot, u64 dataGuid);
void artsSignalEdtPtr(artsGuid_t edtGuid, u32 slot, void * ptr, unsigned int size);

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

artsGuid_t artsGetCurrentEpochGuid();
bool artsSetCurrentEpochGuid(artsGuid_t epochGuid);
artsGuid_t * artsCheckEpochIsRoot(artsGuid_t toCheck);
void artsIncrementFinishedEpochList();

artsGuid_t artsActiveMessageWithDb(artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, artsGuid_t dbGuid);
artsGuid_t artsActiveMessageWithDbAt(artsEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, artsGuid_t dbGuid, unsigned int rank);
artsGuid_t artsActiveMessageWithBuffer(artsEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * ptr, unsigned int size);

typedef struct {
    void * buffer;
    uint32_t * sizeToWrite;
    unsigned int size;
    artsGuid_t epochGuid;
    volatile unsigned int uses;
} artsBuffer_t;

artsGuid_t artsAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, artsGuid_t epochGuid);
void * artsSetBuffer(artsGuid_t bufferGuid, void * buffer, unsigned int size);
void * artsSetBufferNoFree(artsGuid_t bufferGuid, void * buffer, unsigned int size);
void * artsGetBuffer(artsGuid_t bufferGuid);

#ifdef __cplusplus
}
#endif

#endif
