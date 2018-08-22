#ifndef ARTSGLOBALS_H
#define ARTSGLOBALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "artsArrayList.h"
#include "artsCounter.h"
#include "artsQueue.h"

struct atomicCreateBarrierInfo
{
    volatile unsigned int wait;
    volatile unsigned int result;
};

struct artsRuntimeShared
{
    volatile unsigned int sendLock;
    char pad1[56];
    volatile unsigned int recvLock;
    char pad2[56];
    volatile unsigned int stealRequestLock;
    char pad3[56];
    bool (*scheduler)();
    struct artsDeque ** deque;
    struct artsDeque ** receiverDeque;
    artsQueue ** numaQueue;
    struct artsRouteTable ** routeTable;
    struct artsRouteTable * remoteRouteTable;
    volatile bool ** localSpin;
    unsigned int ** memoryMoves;
    struct atomicCreateBarrierInfo ** atomicWaits;
    unsigned int workerThreadCount;
    unsigned int senderThreadCount;
    unsigned int receiverThreadCount;
    unsigned int remoteStealingThreadCount;
    unsigned int totalThreadCount;
    volatile unsigned int readyToPush;
    volatile unsigned int readyToParallelStart;
    volatile unsigned int readyToInspect;
    volatile unsigned int readyToExecute;
    volatile unsigned int readyToClean;
    volatile unsigned int readyToShutdown;
    char * buf;
    int packetSize;
    bool shutdownStarted;
    volatile unsigned int shutdownCount;
    uint64_t shutdownTimeout;
    uint64_t shutdownForceTimeout;
    unsigned int printNodeStats;
    artsGuid_t shutdownEpoch;
    unsigned int shadLoopStride;
}__attribute__ ((aligned(64)));

struct artsRuntimePrivate
{
    struct artsDeque * myDeque;
    struct artsDeque * myNodeDeque;
    unsigned int coreId;
    unsigned int threadId;
    unsigned int groupId;
    unsigned int clusterId;
    unsigned int backOff;
    volatile unsigned int oustandingMemoryMoves;
    struct atomicCreateBarrierInfo atomicWait;
    volatile bool alive;
    volatile bool worker;
    volatile bool networkSend;
    volatile bool networkReceive;
    volatile bool acdt;
    volatile bool statusSend;
    artsGuid_t currentEdtGuid;
    int mallocType;
    int mallocTrace;
    int edtFree;
    int localCounting;
    unsigned int shadLock;
    artsArrayList * counterList;
    unsigned short drand_buf[3];
};

extern struct artsRuntimeShared artsNodeInfo;
extern __thread struct artsRuntimePrivate artsThreadInfo;

extern unsigned int artsGlobalRankId;
extern unsigned int artsGlobalRankCount;
extern unsigned int artsGlobalMasterRankId;
extern bool artsGlobalIWillPrint;
extern uint64_t artsGuidMin;
extern uint64_t artsGuidMax;

#define MASTER_PRINTF(...) if (artsGlobalRankId==artsGlobalMasterRankId) PRINTF(__VA_ARGS__)
#define ONCE_PRINTF(...) if(artsGlobalIWillPrint == true) PRINTF(__VA_ARGS__)

#define artsTypeName const char * const _artsTypeName[] = { \
"ARTS_NULL", \
"ARTS_EDT", \
"ARTS_EVENT", \
"ARTS_EPOCH", \
"ARTS_CALLBACK", \
"ARTS_BUFFER", \
"ARTS_DB_READ", \
"ARTS_DB_WRITE", \
"ARTS_DB_PIN", \
"ARTS_DB_ONCE", \
"ARTS_DB_ONCE_LOCAL", \
"ARTS_LAST_TYPE", \
"ARTS_SINGLE_VALUE", \
"ARTS_PTR" }

#define getTypeName(x) _artsTypeName[x]

extern const char * const _artsTypeName[];

#ifdef __cplusplus
}
#endif

#endif
