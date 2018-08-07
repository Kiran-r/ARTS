#ifndef HIVE_TERMINATION_DETECTION_H
#define  HIVE_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "hive.h"
    
typedef enum
{
    PHASE_1,
    PHASE_2,
    PHASE_3
} TerminationDetectionPhase;
    
typedef struct {
    TerminationDetectionPhase phase;
    volatile unsigned int activeCount;
    volatile unsigned int finishedCount;
    volatile unsigned int globalActiveCount;
    volatile unsigned int globalFinishedCount;
    volatile unsigned int lastActiveCount;
    volatile unsigned int lastFinishedCount;
    volatile u64 queued;
    volatile u64 outstanding;
    unsigned int terminationExitSlot;
    hiveGuid_t terminationExitGuid;
    hiveGuid_t guid;
    hiveGuid_t poolGuid;
    volatile unsigned int * waitPtr;
} hiveEpoch_t;

void incrementQueueEpoch(hiveGuid_t epochGuid);
void incrementActiveEpoch(hiveGuid_t epochGuid);
void incrementFinishedEpoch(hiveGuid_t epochGuid);
void sendEpoch(hiveGuid_t epochGuid, unsigned int source, unsigned int dest);
hiveEpoch_t * createEpoch(hiveGuid_t * guid, hiveGuid_t edtGuid, unsigned int slot);
void hiveAddEdtToEpoch(hiveGuid_t edtGuid, hiveGuid_t epochGuid);
hiveGuid_t hiveInitializeAndStartEpoch(hiveGuid_t finishEdtGuid, unsigned int slot);
hiveGuid_t hiveInitializeEpoch(unsigned int rank, hiveGuid_t finishEdtGuid, unsigned int slot);
void hiveStartEpoch(hiveGuid_t epochGuid);
void broadcastEpochRequest(hiveGuid_t epochGuid);
bool checkEpoch(hiveEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish);
void reduceEpoch(hiveGuid_t epochGuid, unsigned int active, unsigned int finish);
void deleteEpoch(hiveGuid_t epochGuid, hiveEpoch_t * epoch);
bool hiveWaitOnHandle(hiveGuid_t epochGuid);
void hiveYield();

typedef struct hiveEpochPool {
    struct hiveEpochPool * next;
    unsigned int size;
    unsigned int index;
    volatile unsigned int outstanding;
    hiveEpoch_t pool[];
} hiveEpochPool_t;

hiveEpochPool_t * createEpochPool(hiveGuid_t * epochPoolGuid, unsigned int poolSize, hiveGuid_t * startGuid);
hiveEpoch_t * getPoolEpoch(hiveGuid_t edtGuid, unsigned int slot);

void globalShutdownGuidIncActive();
void globalShutdownGuidIncQueue();
void globalShutdownGuidIncFinished();
bool createShutdownEpoch();

#ifdef __cplusplus
}
#endif
#endif
