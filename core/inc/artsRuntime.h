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
