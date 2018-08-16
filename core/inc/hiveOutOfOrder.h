#ifndef HIVEOUTOFORDER_H
#include "hiveOutOfOrderList.h"
#ifdef __cplusplus
extern "C" {
#endif
void hiveOutOfOrderSignalEdt ( hiveGuid_t waitOn, hiveGuid_t edtPacket, hiveGuid_t dataGuid, u32 slot, hiveType_t mode);
void hiveOutOfOrderEventSatisfy( hiveGuid_t waitOn, hiveGuid_t eventGuid, hiveGuid_t dataGuid );
void hiveOutOfOrderEventSatisfySlot( hiveGuid_t waitOn, hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot );
void hiveOutOfOrderAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveType_t mode, hiveGuid_t waitOn);
void hiveOutOfOrderHandleReadyEdt(hiveGuid_t triggerGuid, struct hiveEdt *edt);
void hiveOutOfOrderHandleRemoteDbSend(int rank, hiveGuid_t dbGuid, hiveType_t mode);
void hiveOutOfOrderHandleDbRequestWithOOList(struct hiveOutOfOrderList * addToMe, void ** data, struct hiveEdt *edt, unsigned int slot);
void hiveOutOfOrderHandleDbRequest(hiveGuid_t dbGuid, struct hiveEdt *edt, unsigned int slot);
void hiveOutOfOrderHandleRemoteDbExclusiveRequest(hiveGuid_t dbGuid, int rank, struct hiveEdt * edt, unsigned int slot, hiveType_t mode);
void hiveOutOfOrderHandleRemoteDbFullSend(hiveGuid_t dbGuid, int rank, struct hiveEdt * edt, unsigned int slot, hiveType_t mode);
void hiveOutOfOrderGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void hiveOutOfOrderSignalEdtWithPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void hiveOutOfOrderPutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epcohGuid);
void hiveOutOfOrderIncActiveEpoch(hiveGuid_t epochGuid);
void hiveOutOfOrderIncFinishedEpoch(hiveGuid_t epochGuid);
void hiveOutOfOrderSendEpoch(hiveGuid_t epochGuid, unsigned int source, unsigned int dest);
void hiveOutOfOrderIncQueueEpoch(hiveGuid_t epochGuid);
void hiveOutOfOrderAtomicAddInArrayDb(hiveGuid_t dbGuid,  unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveOutOfOrderAtomicCompareAndSwapInArrayDb(hiveGuid_t dbGuid,  unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveOutOfOrderDbMove(hiveGuid_t dataGuid, unsigned int rank);

void hiveOutOfOrderHandler( void * handleMe, void * memoryPtr );
#ifdef __cplusplus
}
#endif

#endif
