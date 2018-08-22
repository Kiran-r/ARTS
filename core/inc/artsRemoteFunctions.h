#ifndef ARTSEMOTEFUNCTIONS_H
#define ARTSEMOTEFUNCTIONS_H
#include "artsRemoteProtocol.h"
#ifdef __cplusplus
extern "C" {
#endif
void artsRemoteHandleDbLockedRequestReply( void * ptr);
void artsRemoteHandleRequestLockedData( void * ptr);
void artsRemoteRequestLockedData( artsGuid_t dbGuid, unsigned int rank, unsigned int requestingRank );
void artsRemoteHandleDbSignalNext( void * ptr);
void artsRemoteHandleUnlockDb(void * ptr);
void artsRemoteUnlockDb(artsGuid_t dbGuid);
void artsRemoteHandleForwardDbSignalNext(void * ptr);
void artsRemoteForwardDbSignalNext(artsGuid_t dbGuid, unsigned int rank, unsigned int requestingRank);
void artsRemoteDbSignalNext( artsGuid_t dbGuid, void * ptr, unsigned int rank, bool forwarding);
void artsRemoteHandleEdtLockDbRequest(void * ptr);
void artsRemoteEdtLockDbRequest (artsGuid_t dbGuid, unsigned int rank);

int artsExtMemoryMoveAtomicBarrier();
void artsRemoteAddDependence(artsGuid_t source, artsGuid_t destination, u32 slot, artsType_t mode, unsigned int rank);
void artsHandlePingBackInvalidate( void * ptr  );
void artsRemoteHandleUpdateDbGuid( void * ptr  );
void artsRemoteUpdateRouteTablePingBack( void * ptr );
void artsRemoteUpdateRouteTable(artsGuid_t guid, unsigned int rank);
void artsRemoteHandleInvalidateDb(void * ptr);
void artsRemoteHandleMemoryMovePing(void * packet);
void artsRemoteHandleMemoryMoveAtomicPing(void * packet);
void artsRemoteMemoryMove(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*));
void artsRemoteHandleEdtMove( void * ptr  );
void artsRemoteHandleDbMove( void * ptr  );
void artsRemoteHandleMemoryMove( void * ptr  );
void artsRemoteSignalEdt(artsGuid_t edt, artsGuid_t db, u32 slot, artsType_t mode);
void artsRemoteSendStealRequest( unsigned int rank);
void artsRemoteEventSatisfy(artsGuid_t eventGuid, artsGuid_t dataGuid );
void artsRemoteEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, u32 slot );
bool artsRemoteDbRequest(artsGuid_t dataGuid,  int rank, struct artsEdt * edt, int pos, artsType_t mode, bool aggRequest);
void artsRemoteDbSendCheck(int rank, struct artsDb * db, artsType_t mode);
void artsRemoteDbSendNow(int rank, struct artsDb * db);
void artsRemoteDbSend( struct artsRemoteDbRequestPacket * pack );
void artsRemoteHandleDbRecieved( struct artsRemoteDbSendPacket *packet);
bool artsRemoteShutdownSend();
unsigned int packageEdt( void * edtPacket, void ** package );
unsigned int packageEdts( void ** edtPackets, int edtCount, void ** package );
unsigned int packageEdtsAndDbs( void ** edtPackets, int edtCount, void ** package, int rank );
unsigned int handleIncomingEdts( char* address, int edtSizes );
void artsRemotePrintEdtCacheStats();
void artsRemoteDbDestroy( artsGuid_t guid, unsigned int originRank, bool clean );
void artsRemoteHandleDbDestroy( void * ptr );
void artsRemoteHandleDbDestroyForward( void * ptr );
void artsRemoteHandleDbCleanForward( void * ptr );
void artsRemoteDbLock (artsGuid_t dbGuid, void * edtPacket, bool shared);
void artsRemoteDbUnlock (artsGuid_t dbGuid, struct artsDb * db, bool write);
void artsRemoteHandleDbLock(void *ptr);
void artsRemoteHandleDbUnlock(void *ptr);
void artsRemoteHandleDbLockAllDbs(void *ptr);
void artsRemoteDbLockAllDbs(void * edt, unsigned int rank);
void artsRemoteMetricUpdate(int rank, int type, int level, u64 timeStamp, u64 toAdd, bool sub);
void artsRemoteMemoryMoveEmtpyDb(unsigned int route, artsGuid_t guid, void * ptr);
void artsActiveMessage(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize);
void artsRemoteHandleActiveMessage(void * ptr);
void artsRemoteDbForward(int destRank, int sourceRank, artsGuid_t dataGuid, artsType_t mode);
void artsRemoteHandleEventMove(void * ptr);
void artsRemoteHandleUpdateDb(void * ptr);
void artsRemoteUpdateDb(artsGuid_t guid, bool sendDb);
void artsDbRequestCallback(struct artsEdt *edt, unsigned int slot, struct artsDb * dbRes);
void artsRemoteDbFullRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode);
void artsRemoteDbForwardFull(int destRank, int sourceRank, artsGuid_t dataGuid, struct artsEdt * edt, int pos, artsType_t mode);
void artsRemoteDbFullSendNow(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteDbFullSendCheck(int rank, struct artsDb * db, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteDbFullSend(struct artsRemoteDbFullRequestPacket * pack);
void artsRemoteHandleDbFullRecieved(struct artsRemoteDbFullSendPacket * packet);
void artsRemoteSendAlreadyLocal(int rank, artsGuid_t guid, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsRemoteHandleSendAlreadyLocal(void * pack);
void artsRemoteGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, unsigned int rank);
void artsRemoteHandleGetFromDb(void * pack);
void artsRemoteSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void artsRemoteHandleSignalEdtWithPtr(void * pack);
void artsRemotePutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);
void artsRemoteHandlePutInDb(void * pack);
void artsRemoteSend(unsigned int rank, sendHandler_t funPtr, void * args, unsigned int size, bool free);
void artsRemoteHandleSend(void * pack);
void artsRemoteMemoryMoveNoFree(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType);

void artsRemoteEpochInitSend(unsigned int rank, artsGuid_t guid, artsGuid_t edtGuid, unsigned int slot);
void artsRemoteHandleEpochInitSend(void * pack);
void artsRemoteEpochReq(unsigned int rank, artsGuid_t guid);
void artsRemoteHandleEpochReq(void * pack);
void artsRemoteEpochSend(unsigned int rank, artsGuid_t guid, unsigned int active, unsigned int finish);
void artsRemoteHandleEpochSend(void * pack);
void artsRemoteAtomicAddInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicAddInArrayDb(void * pack);
void artsRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack);
void artsRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize, artsGuid_t startGuid, artsGuid_t poolGuid);
void artsRemoteHandleEpochInitPoolSend(void * pack);
void artsRemoteEpochDelete(unsigned int rank, artsGuid_t epochGuid);
void artsRemoteHandleEpochDelete(void * pack);

void artsDbMoveRequest(artsGuid_t dbGuid, unsigned int destRank);
void artsDbMoveRequestHandle(void * pack);

void artsRemoteHandleBufferSend(void * pack);

#ifdef __cplusplus
}
#endif

#endif
