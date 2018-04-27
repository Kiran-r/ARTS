#ifndef HIVERUNTIME_H
#define HIVERUNTIME_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hiveAbstractMachineModel.h"

#define NODEDEQUESIZE 8

enum hiveInitType{ 
    hiveWorkerThread, 
    hiveReceiverThread, 
    hiveRemoteStealThread,
    hiveCounterThread,
    hiveOtherThread
};

void hiveRuntimeNodeInit(unsigned int workerThreads, unsigned int receivingThreads, unsigned int senderThreads, unsigned int receiverThreads, unsigned int totalThreads, bool remoteStealingOn, struct hiveConfig * config);
void hiveRuntimeGlobalCleanup();
void hiveRuntimePrivateCleanup();
void hiveRuntimeStop();
void hiveHandleReadyEdt(struct hiveEdt *edt);
void hiveRehandleReadyEdt(struct hiveEdt *edt);
void hiveHandleReadyEdtNoBlock(struct hiveEdt *edt);
void hiveHandleRemoteStolenEdt(struct hiveEdt *edt);
bool hiveRuntimeSchedulerLoop();
struct hiveEdt * hiveRuntimeStealAnyEdt();
unsigned int hiveRuntimeStealAnyMultipleEdt( unsigned int amount, void ** returnList );
bool hiveRuntimeRemoteBalance();
void hiveThreadZeroNodeStart();
void hiveThreadZeroPrivateInit(struct threadMask * unit, struct hiveConfig * config);
void hiveRuntimePrivateInit(struct threadMask * unit, struct hiveConfig * config);
int hiveRuntimeLoop();
int hiveRuntimeSchedulerLoopWait( volatile bool * waitForMe );

bool hiveRuntimeEdtLockDb (hiveGuid_t dbGuid, struct hiveDb * db, void * edtPacket, bool shared);
void hiveRuntimeEdtLockDbSignalNext (struct hiveDb * db, hiveGuid_t dbGuid, bool remote);
struct hiveEdt * hiveRuntimeStealFromWorker();
struct hiveEdt * hiveRuntimeStealFromNetwork();
void hiveDbUnlock (struct hiveDb * db, hiveGuid_t dbGuid, bool write);
bool hiveDbLockAllDbs( struct hiveEdt * edt );
bool hiveDbLock (hiveGuid_t dbGuid, void * edtPacket, unsigned int rank, bool shared);

bool hiveNetworkFirstSchedulerLoop();
bool hiveNetworkBeforeStealSchedulerLoop();
bool hiveDefaultSchedulerLoop();
#ifdef __cplusplus
}
#endif

#endif
