#ifndef ARTS_TERMINATION_DETECTION_H
#define  ARTS_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"
    
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
    artsGuid_t terminationExitGuid;
    artsGuid_t guid;
    artsGuid_t poolGuid;
    volatile unsigned int * waitPtr;
} artsEpoch_t;

void incrementQueueEpoch(artsGuid_t epochGuid);
void incrementActiveEpoch(artsGuid_t epochGuid);
void incrementFinishedEpoch(artsGuid_t epochGuid);
void sendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
artsEpoch_t * createEpoch(artsGuid_t * guid, artsGuid_t edtGuid, unsigned int slot);
void artsAddEdtToEpoch(artsGuid_t edtGuid, artsGuid_t epochGuid);
artsGuid_t artsInitializeAndStartEpoch(artsGuid_t finishEdtGuid, unsigned int slot);
artsGuid_t artsInitializeEpoch(unsigned int rank, artsGuid_t finishEdtGuid, unsigned int slot);
void artsStartEpoch(artsGuid_t epochGuid);
void broadcastEpochRequest(artsGuid_t epochGuid);
bool checkEpoch(artsEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish);
void reduceEpoch(artsGuid_t epochGuid, unsigned int active, unsigned int finish);
void deleteEpoch(artsGuid_t epochGuid, artsEpoch_t * epoch);
bool artsWaitOnHandle(artsGuid_t epochGuid);
void artsYield();

typedef struct artsEpochPool {
    struct artsEpochPool * next;
    unsigned int size;
    unsigned int index;
    volatile unsigned int outstanding;
    artsEpoch_t pool[];
} artsEpochPool_t;

artsEpochPool_t * createEpochPool(artsGuid_t * epochPoolGuid, unsigned int poolSize, artsGuid_t * startGuid);
artsEpoch_t * getPoolEpoch(artsGuid_t edtGuid, unsigned int slot);

void globalShutdownGuidIncActive();
void globalShutdownGuidIncQueue();
void globalShutdownGuidIncFinished();
bool createShutdownEpoch();

#ifdef __cplusplus
}
#endif
#endif
