#ifndef ARTSINTROSPECTION_H
#define ARTSINTROSPECTION_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsConfig.h"
#include "artsArrayList.h"

#define artsMETRICLEVELS 3
#define artsMAXMETRICNAME 64
#define ARTSSETMEMSHOTTYPE(type) if(artsInternalInspecting()) artsThreadInfo.mallocType =  type
#define artsMEMTRACEON artsThreadInfo.mallocTrace = 1;
#define artsMEMTRACEOFF artsThreadInfo.mallocTrace = 0;

#define artsMETRICNAME const char * const artsMetricName[] = { \
"artsEdtThroughput", \
"artsEdtQueue", \
"artsEdtStealAttempt", \
"artsEdtSteal", \
"artsEdtLastLocalHit", \
"artsEdtSignalThroughput", \
"artsEventSignalThroughput", \
"artsGetBW", \
"artsPutBW", \
"artsNetworkSendBW", \
"artsNetworkRecieveBW", \
"artsNetworkQueuePush", \
"artsNetworkQueuePop", \
"artsYield", \
"artsMallocBW", \
"artsFreeBW", \
"artsRemoteShutdownMsg", \
"artsRemoteEdtStealMsg", \
"artsRemoteEdtRecvMsg", \
"artsRemoteEdtFailMsg", \
"artsRemoteEdtSignalMsg", \
"artsRemoteEventSatisfyMsg", \
"artsRemoteEventSatisfySlotMsg", \
"artsRemoteDbRequestMsg", \
"artsRemoteDbsendMsg", \
"artsRemoteEdtMoveMsg", \
"artsRemoteGuidRouteMsg", \
"artsRemoteEventMoveMsg", \
"artsRemoteAddDependenceMsg", \
"artsRemoteInvalidateDbMsg", \
"artsRemoteDbMoveMsg", \
"artsRemoteDbUpdateGuidMsg", \
"artsRemoteDbUpdateMsg", \
"artsRemoteDbDestroyMsg", \
"artsRemoteDbDestroyForwardMsg", \
"artsRemoteDbCleanForwardMsg", \
"artsRemotePingPongTestMsg", \
"artsDbLockMsg", \
"artsDbUnlockMsg", \
"artsDbLockAllDbsMsg", \
"artsRemoteMetricUpdateMsg", \
"artsActiveMessageMsg", \
"artsRemoteDbFullRequestMsg", \
"artsRemoteDbFullSendMsg", \
"artsRemoteDbFullSendAlreadyLocalMsg", \
"artsRemoteGetFromDbMsg", \
"artsRemotePutInDbMsg", \
"artsRemoteSignalEdtWithPtrMsg", \
"artsRemoteSendMsg", \
"artsEpochInitMsg", \
"artsEpochInitPoolMsg", \
"artsEpochReqMsg", \
"artsEpochSendMsg", \
"artsEpochDeleteMsg", \
"artsAtomicAddArrayDbMsg", \
"artsAtomicCasArrayDbMsg", \
"artsRemoteBufferSendMsg", \
"artsRemoteDbMoveReqMsg", \
"artsDefaultMemorySize", \
"artsEdtMemorySize", \
"artsEventMemorySize", \
"artsDbMemorySize", \
"artsBufferMemorySize", \
"artsDbCount" };

typedef enum artsMetricType 
{
    artsFirstMetricType = -1,
    artsEdtThroughput,
    artsEdtQueue,
    artsEdtStealAttempt,
    artsEdtSteal,
    artsEdtLastLocalHit,
    artsEdtSignalThroughput,
    artsEventSignalThroughput,
    artsGetBW,
    artsPutBW,
    artsNetworkSendBW,
    artsNetworkRecieveBW,
    artsNetworkQueuePush,
    artsNetworkQueuePop,
    artsYieldBW,
    artsMallocBW,
    artsFreeBW,
    artsRemoteShutdownMsg,
    artsRemoteEdtStealMsg,
    artsRemoteEdtRecvMsg,
    artsRemoteEdtFailMsg,
    artsRemoteEdtSignalMsg,
    artsRemoteEventSatisfyMsg,
    artsRemoteEventSatisfySlotMsg,
    artsRemoteDbRequestMsg,
    artsRemoteDbsendMsg,
    artsRemoteEdtMoveMsg,
    artsRemoteGuidRouteMsg,
    artsRemoteEventMoveMsg,
    artsRemoteAddDependenceMsg,
    artsRemoteInvalidateDbMsg,
    artsRemoteDbMoveMsg,
    artsRemoteDbUpdateGuidMsg,
    artsRemoteDbUpdateMsg,
    artsRemoteDbDestroyMsg,
    artsRemoteDbDestroyForwardMsg,
    artsRemoteDbCleanForwardMsg,
    artsRemotePingPongTestMsg,
    artsDbLockMsg,
    artsDbUnlockMsg,
    artsDbLockAllDbsMsg,
    artsRemoteMetricUpdateMsg,
    artsActiveMessageMsg,
    artsRemoteDbFullRequestMsg,
    artsRemoteDbFullSendMsg,
    artsRemoteDbFullSendAlreadyLocalMsg,
    artsRemoteGetFromDbMsg,
    artsRemotePutInDbMsg,
    artsRemoteSignalEdtWithPtrMsg,
    artsRemoteSendMsg, 
    artsEpochInitMsg,
    artsEpochInitPoolMsg,
    artsEpochReqMsg, 
    artsEpochSendMsg,
    artsEpochDeleteMsg,
    artsAtomicAddArrayDbMsg,
    artsAtomicCasArrayDbMsg,
    artsRemoteBufferSendMsg,
    artsRemoteDbMoveReqMsg,
    artsDefaultMemorySize,
    artsEdtMemorySize,
    artsEventMemorySize,
    artsDbMemorySize,
    artsBufferMemorySize,
    artsDbCount,
    artsLastMetricType
} artsMetricType;

typedef enum artsMetricLevel 
{
    artsNoLevel = -1,
    artsThread,
    artsNode,
    artsSystem
} artsMetricLevel;

typedef struct
{
    volatile unsigned int reader;
    char pad1[60];
    volatile unsigned int writer;
    char pad2[60];
    volatile unsigned int intervalReader;
    char pad3[60];
    volatile unsigned int intervalWriter;
    char pad4[60];
    volatile u64 totalBytes;
    volatile u64 totalPackets;
    volatile u64 minPacket;
    volatile u64 maxPacket;
    volatile u64 intervalBytes;
    volatile u64 intervalPackets;
    volatile u64 intervalMin;
    volatile u64 intervalMax;
} artsPacketInspector;

struct artsPerformanceUnit
{
    volatile u64 totalCount;
    char pad1[56];
    volatile u64 maxTotal;
    char pad2[56];
    u64 firstTimeStamp;
    char pad3[56];
    volatile unsigned int lock;
    char pad4[60];
    volatile u64 windowCountStamp;
    volatile u64 windowTimeStamp;
    volatile u64 windowMaxTotal;
    volatile u64 lastWindowCountStamp;
    volatile u64 lastWindowTimeStamp;
    volatile u64 lastWindowMaxTotal;
    u64 (*timeMethod)(void);
}  __attribute__ ((aligned(64)));

typedef struct artsPerformanceUnit artsPerformanceUnit;

typedef struct
{
    unsigned int startPoint;
    u64 startTimeStamp;
    u64 endTimeStamp;
    artsPerformanceUnit * coreMetric;
    artsPerformanceUnit * nodeMetric;
    artsPerformanceUnit * systemMetric;
} artsInspector;

typedef struct
{
    u64 nodeUpdates;
    u64 systemUpdates;
    u64 systemMessages;
    u64 remoteUpdates;
} artsInspectorStats;

typedef struct 
{
    u64 windowCountStamp;
    u64 windowTimeStamp;
    u64 currentCountStamp;
    u64 currentTimeStamp;
    u64 maxTotal;
} artsMetricShot;

typedef struct
{
    artsMetricLevel traceLevel;
    u64 initialStart;
    artsArrayList ** coreMetric;
    artsArrayList ** nodeMetric;
    artsArrayList ** systemMetric;
    unsigned int * nodeLock;
    unsigned int * systemLock;
    char * prefix;
} artsInspectorShots;

void artsInternalReadInspectorConfigFile(char * filename);
void artsInternalStartInspector(unsigned int startPoint);
void artsInternalStopInspector(void);
bool artsInternalInspecting(void);
void artsInternalInitIntrospector(struct artsConfig * config);
u64 artsInternalGetPerformanceMetricTotal(artsMetricType type, artsMetricLevel level);
u64 artsInternalGetPerformanceMetricRateU64(artsMetricType type, artsMetricLevel level, bool last);
u64 artsInternalGetPerformanceMetricRateU64Diff(artsMetricType type, artsMetricLevel level, u64 * diff);
u64 artsInternalGetTotalMetricRateU64(artsMetricType type, artsMetricLevel level, u64 * total, u64 * timeStamp);
double artsInternalGetPerformanceMetricRate(artsMetricType type, artsMetricLevel level, bool last);
bool artsInternalSingleMetricUpdate(artsMetricType type, artsMetricLevel level, u64 *toAdd, bool *sub, artsPerformanceUnit * metric);
void artsInternalHandleRemoteMetricUpdate(artsMetricType type, artsMetricLevel level, u64 toAdd, bool sub);
artsMetricLevel artsInternalUpdatePerformanceMetric(artsMetricType type, artsMetricLevel level, u64 toAdd, bool sub);
artsMetricLevel artsInternalUpdatePerformanceCoreMetric(unsigned int core, artsMetricType type, artsMetricLevel level, u64 toAdd, bool sub);
void artsInternalWriteMetricShotFile(unsigned int threadId, unsigned int nodeId);
void internalPrintTotals(unsigned int nodeId);
void printModelTotalMetrics(artsMetricLevel level);
void artsInternalUpdatePacketInfo(u64 bytes);
void artsInternalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket);
void artsInternalIntervalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket);
void printInspectorStats(void);
void printInspectorTime(void);
void artsInternalSetThreadPerformanceMetric(artsMetricType type, u64 value);
u64 artsGetInspectorTime(void);
double artsInternalGetPerformanceMetricTotalRate(artsMetricType type, artsMetricLevel level);
double artsMetricTest(artsMetricType type, artsMetricLevel level, u64 num);

#ifdef INSPECTOR

#define artsReadInspectorConfigFile(filename) artsInternalReadInspectorConfigFile(filename)
#define artsInitIntrospector(config) artsInternalInitIntrospector(config)
#define artsStartInspector(startPoint) artsInternalStartInspector(startPoint)
#define artsStopInspector() artsInternalStopInspector()
#define artsInspecting() artsInternalInspecting()
#define artsGetPerformanceMetricTotal(type, level) artsInternalGetPerformanceMetricTotal(type, level)
#define artsGetPerformanceMetricRate(type, level, last) artsInternalGetPerformanceMetricRate(type, level, last)
#define artsGetPerformanceMetricTotalRate(type, level) artsInternalGetPerformanceMetricTotalRate(type, level)
#define artsGetPerformanceMetricRateU64(type, level, last) artsInternalGetPerformanceMetricRateU64(type, level, last)
#define artsGetPerformanceMetricRateU64Diff(type, level, diff) artsInternalGetPerformanceMetricRateU64Diff(type, level, diff)
#define artsGetTotalMetricRateU64(type, level, total, timeStamp) artsInternalGetTotalMetricRateU64(type, level, total, timeStamp)
#define artsSingleMetricUpdate(type, level, toAdd, sub, metric) artsInternalSingleMetricUpdate(type, level, toAdd, sub, metric)
#define artsUpdatePerformanceMetric(type, level, toAdd, sub) artsInternalUpdatePerformanceMetric(type, level, toAdd, sub)
#define artsUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) artsInternalUpdatePerformanceCoreMetric(core, type, level, toAdd, sub)
#define artsWriteMetricShotFile(threadId, nodeId) artsInternalWriteMetricShotFile(threadId, nodeId)
#define artsHandleRemoteMetricUpdate(type, level, toAdd, sub) artsInternalHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define artsIntrospectivePrintTotals(nodeId) internalPrintTotals(nodeId)
#define artsUpdatePacketInfo(bytes) artsInternalUpdatePacketInfo(bytes)
#define artsPacketStats(totalBytes, totalPackets, minPacket, maxPacket) artsInternalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket) artsInternalIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsSetThreadPerformanceMetric(type, value) artsInternalSetThreadPerformanceMetric(type, value)

#else

#define artsReadInspectorConfigFile(filename)
#define artsInitIntrospector(config)
#define artsStartInspector(startPoint)
#define artsStopInspector()
#define artsInspecting() 0
#define artsGetPerformanceMetricTotal(type, level) 0
#define artsGetPerformanceMetricRate(type, level, last) 0
#define artsGetPerformanceMetricTotalRate(type, level) 0
#define artsGetPerformanceMetricRateU64(type, level, last) 0
#define artsGetPerformanceMetricRateU64Diff(type, level, diff) 0
#define artsGetTotalMetricRateU64(type, level, total, timeStamp) 0
#define artsSingleMetricUpdate(type, level, toAdd, sub, metric, timeStamp) 0
#define artsUpdatePerformanceMetric(type, level, toAdd, sub) -1
#define artsUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) -1
#define artsWriteMetricShotFile(threadId, nodeId)
#define artsHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define artsIntrospectivePrintTotals(nodeId)
#define artsUpdatePacketInfo(bytes)
#define artsPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define artsSetThreadPerformanceMetric(type, value)

#endif
#ifdef __cplusplus
}
#endif

#endif /* artsINTROSPECTION_H */

