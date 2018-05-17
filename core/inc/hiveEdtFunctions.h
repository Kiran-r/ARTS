#ifndef HIVEEDTFUNCTIONS_H
#define HIVEEDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h>
#define INITIAL_DEPENDENT_SIZE 4

#define hiveEdtArgs(...) sizeof((u64[]){__VA_ARGS__})/sizeof(u64), (u64[]){__VA_ARGS__}
#define hiveEdtEmptySignal(guid) hiveSignalEdt(guid, NULL_GUID, -1, DB_MODE_SINGLE_VALUE);

hiveGuid_t hiveEdtCreate(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc);
hiveGuid_t hiveEdtCreateWithGuid(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc);
hiveGuid_t hiveEdtCreateWithEvent(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc);
hiveGuid_t hiveEdtCreateWithEpoch(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid);
void hiveEdtDelete(struct hiveEdt * edt);
void hiveEdtDestroy(hiveGuid_t guid);
hiveGuid_t hiveEventCreate(unsigned int route, hiveEventTypes_t eventType);
hiveGuid_t hiveEventCreateLatch(unsigned int route, unsigned int latchCount);
hiveGuid_t hiveEventCreateWithGuid(hiveGuid_t guid, hiveEventTypes_t eventType);
hiveGuid_t hiveEventCreateLatchWithGuid(hiveGuid_t guid, unsigned int latchCount);
void hiveEventDestroy(hiveGuid_t guid);
void hiveSignalEdt(hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveDbAccessMode_t mode);
void hiveEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot);
void hiveEventSatisfy(hiveGuid_t eventGuid, hiveGuid_t dataGuid);
void hiveAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode);
void hiveAddLocalEventCallback(hiveGuid_t source, eventCallback_t callback);
bool hiveIsEventFiredExt(hiveGuid_t event);
void hiveSetThreadLocalEdtInfo(struct hiveEdt * edt);
void hiveUnsetThreadLocalEdtInfo();
hiveGuid_t hivePercolateEdt(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t * depv);
void hiveSignalEdtPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, u32 slot);
void hiveRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free);
hiveGuid_t hiveGetCurrentEpochGuid();
bool hiveSetCurrentEpochGuid(hiveGuid_t epochGuid);
hiveGuid_t * hiveCheckEpochIsRoot(hiveGuid_t toCheck);

hiveGuid_t hiveActiveMessageWithDb(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid);
hiveGuid_t hiveActiveMessageWithDbAt(hiveEdt_t funcPtr, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t dbGuid, unsigned int rank);
hiveGuid_t hiveActiveMessageWithBuffer(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, void * ptr, unsigned int size);
#ifdef __cplusplus
}
#endif

hiveGuid_t hiveEdtCreateDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
hiveGuid_t hiveEdtCreateWithGuidDep(hiveEdt_t funcPtr, hiveGuid_t guid, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
hiveGuid_t hiveEdtCreateWithEventDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, bool hasDepv);
hiveGuid_t hiveEdtCreateWithEpochDep(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, u32 depc, hiveGuid_t epochGuid, bool hasDepv);
void hiveSignalEdtNoData(hiveGuid_t edt);

hiveGuid_t hiveEdtCreateShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv);
hiveGuid_t hiveActiveMessageShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, void * data, unsigned int size, hiveGuid_t epochGuid);

typedef struct 
{
    hiveGuid_t currentEdtGuid;
    struct hiveEdt * currentEdt;
    void * epochList;
} threadLocal_t;

void hiveSaveThreadLocal(threadLocal_t * tl);
void hiveRestoreThreadLocal(threadLocal_t * tl);

hiveGuid_t hiveAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses);
void * hiveSetBuffer(hiveGuid_t bufferGuid, void * buffer, unsigned int size);
void * hiveGetBuffer(hiveGuid_t bufferGuid);

#endif
