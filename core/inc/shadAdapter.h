#ifndef SHADADAPTER_H
#define SHADADAPTER_H

#ifdef __cplusplus
extern "C" {
#endif

artsGuid_t artsEdtCreateShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv);
artsGuid_t artsActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size, artsGuid_t epochGuid);
void artsSynchronousActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size);

void artsIncLockShad();
void artsDecLockShad();
void artsCheckLockShad();
void artsStartIntroShad(unsigned int start);
void artsStopIntroShad();
unsigned int artsGetShadLoopStride();

artsGuid_t artsAllocateLocalBufferShad(void ** buffer, uint32_t * sizeToWrite, artsGuid_t epochGuid);

#ifdef __cplusplus
}
#endif

#endif /* SHADADAPTER_H */

