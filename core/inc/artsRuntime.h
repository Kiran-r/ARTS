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
#ifndef ARTSRUNTIME_H
#define ARTSRUNTIME_H
#ifdef __cplusplus
extern "C" {
#endif
#include "artsAbstractMachineModel.h"

#define NODEDEQUESIZE 8

enum artsInitType{ 
    artsWorkerThread, 
    artsReceiverThread, 
    artsRemoteStealThread,
    artsCounterThread,
    artsOtherThread
};

void artsRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct artsConfig * config);
void artsRuntimeGlobalCleanup();
void artsRuntimePrivateCleanup();
void artsRuntimeStop();
void artsHandleReadyEdt(struct artsEdt *edt);
void artsRehandleReadyEdt(struct artsEdt *edt);
void artsHandleReadyEdtNoBlock(struct artsEdt *edt);
void artsHandleRemoteStolenEdt(struct artsEdt *edt);
bool artsRuntimeSchedulerLoop();
struct artsEdt * artsRuntimeStealAnyEdt();
unsigned int artsRuntimeStealAnyMultipleEdt( unsigned int amount, void ** returnList );
bool artsRuntimeRemoteBalance();
void artsThreadZeroNodeStart();
void artsThreadZeroPrivateInit(struct threadMask * unit, struct artsConfig * config);
void artsRuntimePrivateInit(struct threadMask * unit, struct artsConfig * config);
int artsRuntimeLoop();
int artsRuntimeSchedulerLoopWait( volatile bool * waitForMe );

bool artsRuntimeEdtLockDb (artsGuid_t dbGuid, struct artsDb * db, void * edtPacket, bool shared);
void artsRuntimeEdtLockDbSignalNext (struct artsDb * db, artsGuid_t dbGuid, bool remote);
struct artsEdt * artsRuntimeStealFromWorker();
struct artsEdt * artsRuntimeStealFromNetwork();
void artsDbUnlock (struct artsDb * db, artsGuid_t dbGuid, bool write);
bool artsDbLockAllDbs( struct artsEdt * edt );
bool artsDbLock (artsGuid_t dbGuid, void * edtPacket, unsigned int rank, bool shared);

bool artsNetworkFirstSchedulerLoop();
bool artsNetworkBeforeStealSchedulerLoop();
bool artsDefaultSchedulerLoop();
#ifdef __cplusplus
}
#endif

#endif
