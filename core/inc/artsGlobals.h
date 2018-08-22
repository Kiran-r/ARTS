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
    u64 shutdownTimeout;
    u64 shutdownForceTimeout;
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
extern u64 artsGuidMin;
extern u64 artsGuidMax;

#define MASTER_PRINTF(...) if (artsGlobalRankId==artsGlobalMasterRankId) PRINTF(__VA_ARGS__)
#define ONCE_PRINTF(...) if(artsGlobalIWillPrint == true) PRINTF(__VA_ARGS__)
#ifdef __cplusplus
}
#endif

#endif
