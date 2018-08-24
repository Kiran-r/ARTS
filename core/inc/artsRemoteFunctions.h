//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#ifndef ARTSEMOTEFUNCTIONS_H
#define ARTSEMOTEFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsRemoteProtocol.h"

void artsRemoteAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, unsigned int rank);
void artsRemoteUpdateRouteTable(artsGuid_t guid, unsigned int rank);
void artsRemoteHandleUpdateDbGuid(void * ptr);
void artsRemoteHandleInvalidateDb(void * ptr);
void artsRemoteDbDestroy(artsGuid_t guid, unsigned int originRank, bool clean);
void artsRemoteHandleDbDestroyForward(void * ptr);
void artsRemoteHandleDbCleanForward(void * ptr);
void artsRemoteHandleDbDestroy(void * ptr);
void artsRemoteUpdateDb(artsGuid_t guid, bool sendDb);
void artsRemoteHandleUpdateDb(void * ptr);
void artsRemoteMemoryMove(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType, void(*freeMethod)(void*));
void artsRemoteMemoryMoveNoFree(unsigned int route, artsGuid_t guid, void * ptr, unsigned int memSize, unsigned messageType);
void artsRemoteHandleEdtMove(void * ptr);
void artsRemoteHandleDbMove(void * ptr);
void artsRemoteHandleEventMove(void * ptr);
void artsRemoteSignalEdt(artsGuid_t edt, artsGuid_t db, uint32_t slot, artsType_t mode);
void artsRemoteEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot);
void artsDbRequestCallback(struct artsEdt *edt, unsigned int slot, struct artsDb * dbRes);
bool artsRemoteDbRequest(artsGuid_t dataGuid, int rank, struct artsEdt * edt, int pos, artsType_t mode, bool aggRequest);
void artsRemoteDbForward(int destRank, int sourceRank, artsGuid_t dataGuid, artsType_t mode);
void artsRemoteDbSendNow(int rank, struct artsDb * db);
void artsRemoteDbSendCheck(int rank, struct artsDb * db, artsType_t mode);
void artsRemoteDbSend(struct artsRemoteDbRequestPacket * pack);
void artsRemoteHandleDbRecieved(struct artsRemoteDbSendPacket * packet);
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
void artsRemotePutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);
void artsRemoteHandlePutInDb(void * pack);
void artsRemoteSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void artsRemoteHandleSignalEdtWithPtr(void * pack);
bool artsRemoteShutdownSend();
void artsRemoteMetricUpdate(int rank, int type, int level, uint64_t timeStamp, uint64_t toAdd, bool sub);
void artsRemoteHandleSend(void * pack);
void artsRemoteEpochInitSend(unsigned int rank, artsGuid_t epochGuid, artsGuid_t edtGuid, unsigned int slot);
void artsRemoteHandleEpochInitSend(void * pack);
void artsRemoteEpochInitPoolSend(unsigned int rank, unsigned int poolSize, artsGuid_t startGuid, artsGuid_t poolGuid);
void artsRemoteHandleEpochInitPoolSend(void * pack);
void artsRemoteEpochReq(unsigned int rank, artsGuid_t guid);
void artsRemoteHandleEpochReq(void * pack);
void artsRemoteEpochSend(unsigned int rank, artsGuid_t guid, unsigned int active, unsigned int finish);
void artsRemoteHandleEpochSend(void * pack);
void artsRemoteAtomicAddInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicAddInArrayDb(void * pack);
void artsRemoteAtomicCompareAndSwapInArrayDb(unsigned int rank, artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsRemoteHandleAtomicCompareAndSwapInArrayDb(void * pack);
void artsRemoteEpochDelete(unsigned int rank, artsGuid_t epochGuid);
void artsRemoteHandleEpochDelete(void * pack);
void artsDbMoveRequest(artsGuid_t dbGuid, unsigned int destRank);
void artsDbMoveRequestHandle(void * pack);
void artsRemoteHandleBufferSend(void * pack);
void artsRemoteHandleDbDestroy(void * ptr);

#ifdef __cplusplus
}
#endif

#endif
