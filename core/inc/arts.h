#ifndef ARTS_H
#define ARTS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRT.h"
    
int artsRT(int argc, char **argv);
void artsShutdown();

/*Malloc***********************************************************************/

void *artsMalloc(size_t size);
void *artsMallocAlign(size_t size, size_t align);
void *artsCalloc(size_t size);
void *artsCallocAlign(size_t size, size_t allign);
void * artsRealloc(void *ptr, size_t size);
void artsFree(void *ptr);
void artsFreeAlign(void *ptr);

/*GUID*************************************************************************/

artsGuid_t artsReserveGuidRoute(artsType_t type, unsigned int route);
artsGuid_t artsReserveGuidRouteRemote(artsType_t type, unsigned int route);
bool artsIsGuidLocal(artsGuid_t guid);
unsigned int artsGuidGetRank(artsGuid_t guid);
artsType_t artsGuidGetType(artsGuid_t guid);
artsGuid_t artsGuidCast(artsGuid_t guid, artsType_t type);

artsGuidRange * artsNewGuidRangeNode(unsigned int type, unsigned int size, unsigned int route);
artsGuid_t artsGetGuid(artsGuidRange * range, unsigned int index);
artsGuid_t artsGuidRangeNext(artsGuidRange * range);
bool artsGuidRangeHasNext(artsGuidRange * range);
void artsGuidRangeResetIter(artsGuidRange * range);

/*EDT**************************************************************************/
    
artsGuid_t artsEdtCreate(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc);
artsGuid_t artsEdtCreateWithGuid(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc);
artsGuid_t artsEdtCreateWithEpoch(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t epochGuid);

artsGuid_t artsEdtCreateDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool hasDepv);
artsGuid_t artsEdtCreateWithGuidDep(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool hasDepv);
artsGuid_t artsEdtCreateWithEpochDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t epochGuid, bool hasDepv);

void artsEdtDestroy(artsGuid_t guid);

void artsSignalEdt(artsGuid_t edtGuid, uint32_t slot, artsGuid_t dataGuid);
void artsSignalEdtValue(artsGuid_t edtGuid, uint32_t slot, uint64_t dataGuid);
void artsSignalEdtPtr(artsGuid_t edtGuid, uint32_t slot, void * ptr, unsigned int size);

artsGuid_t artsActiveMessageWithDb(artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t dbGuid);
artsGuid_t artsActiveMessageWithDbAt(artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsGuid_t dbGuid, unsigned int rank);
artsGuid_t artsActiveMessageWithBuffer(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, void * ptr, unsigned int size);

artsGuid_t artsAllocateLocalBuffer(void ** buffer, unsigned int size, unsigned int uses, artsGuid_t epochGuid);
void * artsSetBuffer(artsGuid_t bufferGuid, void * buffer, unsigned int size);
void * artsSetBufferNoFree(artsGuid_t bufferGuid, void * buffer, unsigned int size);
void * artsGetBuffer(artsGuid_t bufferGuid);

/*Event************************************************************************/

artsGuid_t artsEventCreate(unsigned int route, unsigned int latchCount);
artsGuid_t artsEventCreateWithGuid(artsGuid_t guid, unsigned int latchCount);

bool artsIsEventFiredExt(artsGuid_t event);
void artsEventDestroy(artsGuid_t guid);

void artsEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot);

void artsAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot);
void artsAddLocalEventCallback(artsGuid_t source, eventCallback_t callback);

/*DB***************************************************************************/

artsGuid_t artsDbCreate(void **addr, uint64_t size, artsType_t mode);
void * artsDbCreateWithGuid(artsGuid_t guid, uint64_t size);
void * artsDbCreateWithGuidAndData(artsGuid_t guid, void * data, uint64_t size);
artsGuid_t artsDbCreateRemote(unsigned int route, uint64_t size, artsType_t mode);

void * artsDbResize(artsGuid_t guid, unsigned int size, bool copy);
void artsDbMove(artsGuid_t dbGuid, unsigned int rank);

void artsDbDestroy(artsGuid_t guid);
void artsDbDestroySafe(artsGuid_t guid, bool remote);

void artsPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsPutInDbAt(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void artsPutInDbEpoch(void * ptr, artsGuid_t epochGuid, artsGuid_t dbGuid, unsigned int offset, unsigned int size);

void artsGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsGetFromDbAt(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);

/*Epoch************************************************************************/

artsGuid_t artsGetCurrentEpochGuid();
void artsAddEdtToEpoch(artsGuid_t edtGuid, artsGuid_t epochGuid);
artsGuid_t artsInitializeAndStartEpoch(artsGuid_t finishEdtGuid, unsigned int slot);
artsGuid_t artsInitializeEpoch(unsigned int rank, artsGuid_t finishEdtGuid, unsigned int slot);
void artsStartEpoch(artsGuid_t epochGuid);
bool artsWaitOnHandle(artsGuid_t epochGuid);
void artsYield();

/*ArrayDb************************************************************************/

unsigned int artsGetSizeArrayDb(artsArrayDb_t * array);
artsGuid_t artsNewArrayDb(artsArrayDb_t **addr, unsigned int elementSize, unsigned int numElements);
artsArrayDb_t * artsNewArrayDbWithGuid(artsGuid_t guid, unsigned int elementSize, unsigned int numElements);
void artsGetFromArrayDb(artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);
void artsPutInArrayDb(void * ptr, artsGuid_t edtGuid, unsigned int slot, artsArrayDb_t * array, unsigned int index);
void artsForEachInArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv);
void artsForEachInArrayDbAtData(artsArrayDb_t * array, unsigned int stride, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv);
void artsGatherArrayDb(artsArrayDb_t * array, artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint64_t depc);
void artsAtomicAddInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot);
void artsAtomicCompareAndSwapInArrayDb(artsArrayDb_t * array, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot);

/*Util*************************************************************************/

artsGuid_t artsGetCurrentGuid();
unsigned int artsGetCurrentNode();
unsigned int artsGetTotalNodes();
unsigned int artsGetCurrentWorker();
unsigned int artsGetTotalWorkers();
unsigned int artsGetCurrentCluster();
unsigned int artsGetTotalClusters();
uint64_t artsGetTimeStamp();

#ifdef __cplusplus
}
#endif
#endif
