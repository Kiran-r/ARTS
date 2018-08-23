#ifndef ARTS_TERMINATION_DETECTION_H
#define  ARTS_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"

artsEpoch_t * createEpoch(artsGuid_t * guid, artsGuid_t edtGuid, unsigned int slot);
void incrementQueueEpoch(artsGuid_t epochGuid);
void incrementActiveEpoch(artsGuid_t epochGuid);
void incrementFinishedEpoch(artsGuid_t epochGuid);
void sendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
void broadcastEpochRequest(artsGuid_t epochGuid);
bool checkEpoch(artsEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish);
void reduceEpoch(artsGuid_t epochGuid, unsigned int active, unsigned int finish);
void deleteEpoch(artsGuid_t epochGuid, artsEpoch_t * epoch);

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
