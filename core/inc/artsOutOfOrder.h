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
#ifndef ARTSOUTOFORDER_H
#define ARTSOUTOFORDER_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "artsOutOfOrderList.h"
void artsOutOfOrderSignalEdt ( artsGuid_t waitOn, artsGuid_t edtPacket, artsGuid_t dataGuid, uint32_t slot, artsType_t mode);
void artsOutOfOrderEventSatisfy( artsGuid_t waitOn, artsGuid_t eventGuid, artsGuid_t dataGuid );
void artsOutOfOrderEventSatisfySlot( artsGuid_t waitOn, artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot );
void artsOutOfOrderAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot, artsType_t mode, artsGuid_t waitOn);
void artsOutOfOrderHandleReadyEdt(artsGuid_t triggerGuid, struct artsEdt *edt);
void artsOutOfOrderHandleRemoteDbSend(int rank, artsGuid_t dbGuid, artsType_t mode);
void artsOutOfOrderHandleDbRequestWithOOList(struct artsOutOfOrderList * addToMe, void ** data, struct artsEdt *edt, unsigned int slot);
void artsOutOfOrderHandleDbRequest(artsGuid_t dbGuid, struct artsEdt *edt, unsigned int slot);
void artsOutOfOrderHandleRemoteDbExclusiveRequest(artsGuid_t dbGuid, int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsOutOfOrderHandleRemoteDbFullSend(artsGuid_t dbGuid, int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void artsOutOfOrderGetFromDb(artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size);
void artsOutOfOrderSignalEdtWithPtr(artsGuid_t edtGuid, artsGuid_t dbGuid, void * ptr, unsigned int size, unsigned int slot);
void artsOutOfOrderPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epcohGuid);
void artsOutOfOrderIncActiveEpoch(artsGuid_t epochGuid);
void artsOutOfOrderIncFinishedEpoch(artsGuid_t epochGuid);
void artsOutOfOrderSendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
void artsOutOfOrderIncQueueEpoch(artsGuid_t epochGuid);
void artsOutOfOrderAtomicAddInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsOutOfOrderAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid,  unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void artsOutOfOrderDbMove(artsGuid_t dataGuid, unsigned int rank);

void artsOutOfOrderHandler( void * handleMe, void * memoryPtr );

#ifdef __cplusplus
}
#endif

#endif
