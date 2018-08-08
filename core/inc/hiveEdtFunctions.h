#ifndef HIVEEDTFUNCTIONS_H
#define HIVEEDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h>

#define hiveEdtArgs(...) sizeof((u64[]){__VA_ARGS__})/sizeof(u64), (u64[]){__VA_ARGS__}
#define hiveEdtEmptySignal(guid) hiveSignalEdt(guid, NULL_GUID, -1, DB_MODE_SINGLE_VALUE);

bool hiveEdtCreateInternal(hiveGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, hiveGuid_t eventGuid, hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, bool useEpoch, hiveGuid_t epochGuid, bool hasDepv);

hiveGuid_t hiveEdtCreate(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc);
hiveGuid_t hiveEdtCreateWithGuid(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc);
hiveGuid_t hiveEdtCreateWithEpoch(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid);

hiveGuid_t hiveEdtCreateDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
hiveGuid_t hiveEdtCreateWithGuidDep(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
hiveGuid_t hiveEdtCreateWithEpochDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid, bool hasDepv);

void hiveEdtDelete(struct hiveEdt * edt);
void hiveEdtDestroy(hiveGuid_t guid);

void hiveSignalEdt(hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveDbAccessMode_t mode);
void hiveSignalEdtPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, u32 slot);

typedef struct 
{
    hiveGuid_t currentEdtGuid;
    struct hiveEdt * currentEdt;
    void * epochList;
} threadLocal_t;

void hiveSetThreadLocalEdtInfo(struct hiveEdt * edt);
void hiveUnsetThreadLocalEdtInfo();
void hiveSaveThreadLocal(threadLocal_t * tl);
void hiveRestoreThreadLocal(threadLocal_t * tl);

hiveGuid_t hiveGetCurrentEpochGuid();
bool hiveSetCurrentEpochGuid(hiveGuid_t epochGuid);
hiveGuid_t * hiveCheckEpochIsRoot(hiveGuid_t toCheck);
void hiveIncrementFinishedEpochList();

hiveGuid_t hiveActiveMessageWithDb(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid);
hiveGuid_t hiveActiveMessageWithDbAt(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid, unsigned int rank);
hiveGuid_t hiveActiveMessageWithBuffer(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * ptr, unsigned int size);

typedef struct {
    void * buffer;
    uint32_t * sizeToWrite;
    unsigned int size;
    hiveGuid_t epochGuid;
    volatile unsigned int uses;
} hiveBuffer_t;

hiveGuid_t hiveAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, hiveGuid_t epochGuid);
void * hiveSetBuffer(hiveGuid_t bufferGuid, void * buffer, unsigned int size);
void * hiveSetBufferNoFree(hiveGuid_t bufferGuid, void * buffer, unsigned int size);
void * hiveGetBuffer(hiveGuid_t bufferGuid);

#ifdef __cplusplus
}
#endif

#endif
