#ifndef SHADADAPTER_H
#define SHADADAPTER_H

#ifdef __cplusplus
extern "C" {
#endif

hiveGuid_t hiveEdtCreateShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv);
hiveGuid_t hiveActiveMessageShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, void * data, unsigned int size, hiveGuid_t epochGuid);
void hiveSynchronousActiveMessageShad(hiveEdt_t funcPtr, unsigned int route, u32 paramc, u64 * paramv, void * data, unsigned int size);

void hiveIncLockShad();
void hiveDecLockShad();
void hiveCheckLockShad();
void hiveStartIntroShad(unsigned int start);
void hiveStopIntroShad();
unsigned int hiveGetShadLoopStride();

hiveGuid_t hiveAllocateLocalBufferShad(void ** buffer, uint32_t * sizeToWrite, hiveGuid_t epochGuid);

#ifdef __cplusplus
}
#endif

#endif /* SHADADAPTER_H */

