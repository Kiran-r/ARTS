#ifndef HIVEGLOBALS_H
#define HIVEGLOBALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "hive.h"
#include "hiveArrayList.h"
#include "hiveCounter.h"

struct atomicCreateBarrierInfo
{
    volatile unsigned int wait;
    volatile unsigned int result;
};

struct hiveRuntimeShared
{
    volatile unsigned int sendLock;
    char pad1[56];
    volatile unsigned int recvLock;
    char pad2[56];
    volatile unsigned int stealRequestLock;
    char pad3[56];
    bool (*scheduler)();
    struct hiveDeque ** deque;
    struct hiveDeque ** nodeDeque;
    struct hiveDeque ** workerDeque;
    struct hiveDeque ** workerNodeDeque;
    struct hiveDeque ** receiverDeque;
    struct hiveDeque ** receiverNodeDeque;
    struct hiveRouteTable ** routeTable;
    struct hiveRouteTable * remoteRouteTable;
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
}__attribute__ ((aligned(64)));;

struct hiveRuntimePrivate
{
    struct hiveDeque * myDeque;
    struct hiveDeque * myNodeDeque;
    struct hiveRouteTable *  myRouteTable;
    unsigned int coreId;
    unsigned int threadId;
    unsigned int groupId;
    unsigned int backOff;
    volatile unsigned int oustandingMemoryMoves;
    struct atomicCreateBarrierInfo atomicWait;
    volatile bool alive;
    volatile bool worker;
    volatile bool networkSend;
    volatile bool networkReceive;
    volatile bool acdt;
    volatile bool statusSend;
    hiveGuid_t currentEdtGuid;
    int mallocType;
    int mallocTrace;
    int edtFree;
    int localCounting;
    hiveArrayList * counterList;
    unsigned short drand_buf[3];
};

extern struct hiveRuntimeShared hiveNodeInfo;
extern __thread struct hiveRuntimePrivate hiveThreadInfo;

extern unsigned int hiveGlobalRankId;
extern unsigned int hiveGlobalRankCount;
extern unsigned int hiveGlobalMasterRankId;
extern bool hiveGlobalIWillPrint;
extern u64 hiveGuidMin;
extern u64 hiveGuidMax;

#define MASTER_PRINTF(...) if (hiveGlobalRankId==hiveGlobalMasterRankId) PRINTF(__VA_ARGS__)
#define ONCE_PRINTF(...) if(hiveGlobalIWillPrint == true) PRINTF(__VA_ARGS__)
#ifdef __cplusplus
}
#endif

#endif
