#ifndef HIVEEMOTEFUNCTIONS_H
#define HIVEEMOTEFUNCTIONS_H
#include "hiveRemoteProtocol.h"
#ifdef __cplusplus
extern "C" {
#endif
void hiveRemoteHandleDbLockedRequestReply( void * ptr);
void hiveRemoteHandleRequestLockedData( void * ptr);
void hiveRemoteRequestLockedData( hiveGuid_t dbGuid, unsigned int rank, unsigned int requestingRank );
void hiveRemoteHandleDbSignalNext( void * ptr);
void hiveRemoteHandleUnlockDb(void * ptr);
void hiveRemoteUnlockDb(hiveGuid_t dbGuid);
void hiveRemoteHandleForwardDbSignalNext(void * ptr);
void hiveRemoteForwardDbSignalNext(hiveGuid_t dbGuid, unsigned int rank, unsigned int requestingRank);
void hiveRemoteDbSignalNext( hiveGuid_t dbGuid, void * ptr, unsigned int rank, bool forwarding);
void hiveRemoteHandleEdtLockDbRequest(void * ptr);
void hiveRemoteEdtLockDbRequest (hiveGuid_t dbGuid, unsigned int rank);

int hiveExtMemoryMoveAtomicBarrier();
void hiveRemoteAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode, unsigned int rank);
void hiveHandlePingBackInvalidate( void * ptr  );
void hiveRemoteHandleUpdateDbGuid( void * ptr  );
void hiveRemoteUpdateRouteTablePingBack( void * ptr );
void hiveRemoteUpdateRouteTable(hiveGuid_t guid, unsigned int rank);
void hiveRemoteHandleInvalidateDb(void * ptr);
void hiveRemoteHandleMemoryMovePing(void * packet);
void hiveRemoteHandleMemoryMoveAtomicPing(void * packet);
void hiveRemoteMemoryMove(unsigned int route, hiveGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*));
void hiveRemoteHandleEdtMove( void * ptr  );
void hiveRemoteHandleDbMove( void * ptr  );
void hiveRemoteHandleMemoryMove( void * ptr  );
void hiveRemoteSignalEdt(hiveGuid_t edt, hiveGuid_t db, u32 slot, hiveDbAccessMode_t mode);
void hiveRemoteSendStealRequest( unsigned int rank);
void hiveRemoteEventSatisfy(hiveGuid_t eventGuid, hiveGuid_t dataGuid );
void hiveRemoteEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot );
bool hiveRemoteDbRequest(hiveGuid_t dataGuid,  int rank, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode, bool aggRequest);
void hiveRemoteDbSendCheck(int rank, struct hiveDb * db, hiveDbAccessMode_t mode);
void hiveRemoteDbSendNow(int rank, struct hiveDb * db);
void hiveRemoteDbSend( struct hiveRemoteDbRequestPacket * pack );
void hiveRemoteHandleDbRecieved( struct hiveRemoteDbSendPacket *packet);
bool hiveRemoteShutdownSend();
unsigned int packageEdt( void * edtPacket, void ** package );
unsigned int packageEdts( void ** edtPackets, int edtCount, void ** package );
unsigned int packageEdtsAndDbs( void ** edtPackets, int edtCount, void ** package, int rank );
unsigned int handleIncomingEdts( char* address, int edtSizes );
void handleIncomingEdtsAndDbs( char* address, int edtSizes );
void hiveRemotePrintEdtCacheStats();
void hiveRemoteDbDestroy( hiveGuid_t guid, unsigned int originRank, bool clean );
void hiveRemoteHandleDbDestroy( void * ptr );
void hiveRemoteHandleDbDestroyForward( void * ptr );
void hiveRemoteHandleDbCleanForward( void * ptr );
void hiveRemoteDbLock (hiveGuid_t dbGuid, void * edtPacket, bool shared);
void hiveRemoteDbUnlock (hiveGuid_t dbGuid, struct hiveDb * db, bool write);
void hiveRemoteHandleDbLock(void *ptr);
void hiveRemoteHandleDbUnlock(void *ptr);
void hiveRemoteHandleDbLockAllDbs(void *ptr);
void hiveRemoteDbLockAllDbs(void * edt, unsigned int rank);
void hiveRemoteMetricUpdate(int rank, int type, int level, u64 timeStamp, u64 toAdd, bool sub);
void hiveRemoteMemoryMoveEmtpyDb(unsigned int route, hiveGuid_t guid, void * ptr);
void hiveActiveMessage(unsigned int route, hiveGuid_t guid, void * ptr, unsigned int memSize);
void hiveRemoteHandleActiveMessage(void * ptr);
void hiveRemoteDbForward(int destRank, int sourceRank, hiveGuid_t dataGuid, hiveDbAccessMode_t mode);
void hiveRemoteHandleEventMove(void * ptr);
void hiveRemoteHandleUpdateDb(void * ptr);
void hiveRemoteUpdateDb(hiveGuid_t guid, bool sendDb);
void hiveDbRequestCallback(struct hiveEdt *edt, unsigned int slot, struct hiveDb * dbRes);
void hiveRemoteDbFullRequest(hiveGuid_t dataGuid, int rank, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode);
void hiveRemoteDbForwardFull(int destRank, int sourceRank, hiveGuid_t dataGuid, struct hiveEdt * edt, int pos, hiveDbAccessMode_t mode);
void hiveRemoteDbFullSendNow(int rank, struct hiveDb * db, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
void hiveRemoteDbFullSendCheck(int rank, struct hiveDb * db, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
void hiveRemoteDbFullSend(struct hiveRemoteDbFullRequestPacket * pack);
void hiveRemoteHandleDbFullRecieved(struct hiveRemoteDbFullSendPacket * packet);
void hiveRemoteSendAlreadyLocal(int rank, hiveGuid_t guid, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
void hiveRemoteHandleSendAlreadyLocal(void * pack);
void hiveRemoteGetFromDb(hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void hiveRemoteHandleGetFromDb(void * pack);
void hiveRemoteSignalEdtWithPtr(hiveGuid_t edtGuid, hiveGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void hiveRemoteHandleSignalEdtWithPtr(void * pack);
void hiveRemotePutInDb(void * ptr, hiveGuid_t edtGuid, hiveGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, hiveGuid_t epochGuid, unsigned int rank);
void hiveRemoteHandlePutInDb(void * pack);
void hiveRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free);
void hiveRemoteHandleSend(void * pack);
void hiveRemoteMemoryMoveNoFree(unsigned int route, hiveGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType);

void hiveRemoteEpochInitSend(unsigned int rank, hiveGuid_t guid, hiveGuid_t edtGuid, unsigned int slot);
void hiveRemoteHandleEpochInitSend(void * pack);
void hiveRemoteEpochReq(unsigned int rank, hiveGuid_t guid);
void hiveRemoteHandleEpochReq(void * pack);
void hiveRemoteEpochSend(unsigned int rank, hiveGuid_t guid, unsigned int active, unsigned int finish);
void hiveRemoteHandleEpochSend(void * pack);
void hiveRemoteAtomicAddInArrayDb(unsigned int rank, hiveGuid_t dbGuid, unsigned int index, unsigned int toAdd, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveRemoteHandleAtomicAddInArrayDb(void * pack);
void hiveRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, hiveGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, hiveGuid_t edtGuid, unsigned int slot, hiveGuid_t epochGuid);
void hiveRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack);
void hiveRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize, hiveGuid_t startGuid, hiveGuid_t poolGuid);
void hiveRemoteHandleEpochInitPoolSend(void * pack);
void hiveRemoteEpochDelete(unsigned int rank, hiveGuid_t epochGuid);
void hiveRemoteHandleEpochDelete(void * pack);

void hiveDbMoveRequest(hiveGuid_t dbGuid, unsigned int destRank);
void hiveDbMoveRequestHandle(void * pack);

void hiveRemoteHandleBufferSend(void * pack);

#ifdef __cplusplus
}
#endif

#endif
