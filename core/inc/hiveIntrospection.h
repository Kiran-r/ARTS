#ifndef HIVEINTROSPECTION_H
#define HIVEINTROSPECTION_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hive.h"
#include "hiveConfig.h"
#include "hiveArrayList.h"

#define hiveMETRICLEVELS 3
#define hiveMAXMETRICNAME 64
#define HIVESETMEMSHOTTYPE(type) if(hiveInternalInspecting()) hiveThreadInfo.mallocType =  type
#define hiveMEMTRACEON hiveThreadInfo.mallocTrace = 1;
#define hiveMEMTRACEOFF hiveThreadInfo.mallocTrace = 0;

#define hiveMETRICNAME const char * const hiveMetricName[] = { \
"hiveEdtThroughput", \
"hiveEdtSignalThroughput", \
"hiveEventSignalThroughput", \
"hiveNetworkSendBW", \
"hiveNetworkRecieveBW", \
"hiveNetworkQueuePush", \
"hiveNetworkQueuePop", \
"hiveRemoteEdtSteal", \
"hiveMallocBW", \
"hiveFreeBW", \
"hiveIOBandwidth", \
"hiveCompression", \
"hiveIOKeyThroughput", \
"hiveKeyThroughput", \
"hiveLocalReduceThroughput", \
"hiveShuffleThroughput", \
"hiveReduceThroughput", \
"hiveRemoteEdtStealMsg", \
"hiveRemoteEdtRecvMsg", \
"hiveRemoteEdtFailMsg", \
"hiveRemoteEdtSignalMsg", \
"hiveRemoteEventSatisfyMsg", \
"hiveRemoteEventSatisfySlotMsg", \
"hiveRemoteDbRequestMsg", \
"hiveRemoteDbSendMsg", \
"hiveRemoteEdtMoveMsg", \
"hiveRemoteGuidRouteMsg", \
"hiveRemoteMemoryMoveMsg", \
"hiveRemoteAddDependenceMsg", \
"hiveRemoteAcdtCompositeNewMsg", \
"hiveRemoteInvalidateDbMsg", \
"hiveRemoteDbMoveMsg", \
"hiveRemoteDbUpdateGuidMsg", \
"hiveRemoteDbUpdateGuidPingMsg", \
"hiveRemoteInvalidateDbPingMsg", \
"hiveRemoteTemplateRequestMsg", \
"hiveRemoteTemplateSendMsg", \
"hiveRemoteMemoryMovePingMsg", \
"hiveRemoteMemoryMoveAtomicPingMsg", \
"hiveRemoteDbUpdateMsg", \
"hiveRemoteDbDestroyMsg", \
"hiveRemoteDbDestroyForwardMsg", \
"hiveRemoteDbCleanForwardMsg", \
"hiveRemoteTaskSignalMsg", \
"hiveRemoteStreamSignalMsg", \
"hiveRemoteStreamRegisterMsg", \
"hiveRemoteStreamWriteMsg", \
"hiveRemotePingpongTestMsg", \
"hiveDbLockMsg", \
"hiveDbUnlockMsg", \
"hiveDbLockAllDbsMsg", \
"hiveDbLockAllDbsIntelMsg", \
"hiveDbLocalReduction", \
"hiveShuffleDone", \
"hiveDefaultMemorySize", \
"hiveEdtMemorySize", \
"hiveEventMemorySize", \
"hiveDbMemorySize", \
"hiveMapperThreshold", \
"hiveMapperBlockSize", \
"hiveLocalReducerBlockSize", \
"hiveDbCount" };

typedef enum hiveMetricType 
{
    hiveFirstMetricType = -1,
    hiveEdtThroughput,
    hiveEdtSignalThroughput,
    hiveEventSignalThroughput,
    hiveNetworkSendBW,
    hiveNetworkRecieveBW,
    hiveNetworkQueuePush,
    hiveNetworkQueuePop,
    hiveRemoteEdtSteal,
    hiveMallocBW,
    hiveFreeBW,
    hiveIOBandwidth,
    hiveCompression,
    hiveIOKeyThroughput,
    hiveKeyThroughput,
    hiveLocalReduceThroughput,
    hiveShuffleThroughput,
    hiveReduceThroughput,
    hiveRemoteEdtStealMsg,
    hiveRemoteEdtRecvMsg,
    hiveRemoteEdtFailMsg,
    hiveRemoteEdtSignalMsg,
    hiveRemoteEventSatisfyMsg,
    hiveRemoteEventSatisfySlotMsg,
    hiveRemoteDbRequestMsg,
    hiveRemoteDbSendMsg,
    hiveRemoteEdtMoveMsg,
    hiveRemoteGuidRouteMsg,
    hiveRemoteMemoryMoveMsg,
    hiveRemoteAddDependenceMsg,
    hiveRemoteAcdtCompositeNewMsg,
    hiveRemoteInvalidateDbMsg,
    hiveRemoteDbMoveMsg,
    hiveRemoteDbUpdateGuidMsg,
    hiveRemoteDbUpdateGuidPingMsg,
    hiveRemoteInvalidateDbPingMsg,
    hiveRemoteTemplateRequestMsg,
    hiveRemoteTemplateSendMsg,
    hiveRemoteMemoryMovePingMsg,
    hiveRemoteMemoryMoveAtomicPingMsg,
    hiveRemoteDbUpdateMsg,
    hiveRemoteDbDestroyMsg,
    hiveRemoteDbDestroyForwardMsg,
    hiveRemoteDbCleanForwardMsg,
    hiveRemoteTaskSignalMsg,
    hiveRemoteStreamSignalMsg,
    hiveRemoteStreamRegisterMsg,
    hiveRemoteStreamWriteMsg,
    hiveRemotePingpongTestMsg,
    hiveDbLockMsg,
    hiveDbUnlockMsg,
    hiveDbLockAllDbsMsg,
    hiveDbLockAllDbsIntelMsg,
    hiveDbLocalReduction,
    hiveShuffleDone,
    hiveDefaultMemorySize,
    hiveEdtMemorySize,
    hiveEventMemorySize,
    hiveDbMemorySize,
    hiveMapperThreshold,
    hiveMapperBlockSize,
    hiveLocalReducerBlockSize,
    hiveDbCount,
    hiveLastMetricType
} hiveMetricType;

typedef enum hiveMetricLevel 
{
    hiveNoLevel = -1,
    hiveThread,
    hiveNode,
    hiveSystem
} hiveMetricLevel;

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
} hivePacketInspector;

struct hivePerformanceUnit
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

typedef struct hivePerformanceUnit hivePerformanceUnit;

typedef struct
{
    unsigned int startPoint;
    u64 startTimeStamp;
    u64 endTimeStamp;
    hivePerformanceUnit * coreMetric;
    hivePerformanceUnit * nodeMetric;
    hivePerformanceUnit * systemMetric;
} hiveInspector;

typedef struct
{
    u64 nodeUpdates;
    u64 systemUpdates;
    u64 systemMessages;
    u64 remoteUpdates;
} hiveInspectorStats;

typedef struct 
{
    u64 windowCountStamp;
    u64 windowTimeStamp;
    u64 currentCountStamp;
    u64 currentTimeStamp;
    u64 maxTotal;
} hiveMetricShot;

typedef struct
{
    hiveMetricLevel traceLevel;
    u64 initialStart;
    hiveArrayList ** coreMetric;
    hiveArrayList ** nodeMetric;
    hiveArrayList ** systemMetric;
    unsigned int * nodeLock;
    unsigned int * systemLock;
    char * prefix;
} hiveInspectorShots;

void hiveInternalReadInspectorConfigFile(char * filename);
void hiveInternalStartInspector(unsigned int startPoint);
void hiveInternalStopInspector(void);
bool hiveInternalInspecting(void);
void hiveInternalInitIntrospector(struct hiveConfig * config);
u64 hiveInternalGetPerformanceMetricTotal(hiveMetricType type, hiveMetricLevel level);
u64 hiveInternalGetPerformanceMetricRateU64(hiveMetricType type, hiveMetricLevel level, bool last);
u64 hiveInternalGetPerformanceMetricRateU64Diff(hiveMetricType type, hiveMetricLevel level, u64 * diff);
u64 hiveInternalGetTotalMetricRateU64(hiveMetricType type, hiveMetricLevel level, u64 * total, u64 * timeStamp);
double hiveInternalGetPerformanceMetricRate(hiveMetricType type, hiveMetricLevel level, bool last);
bool hiveInternalSingleMetricUpdate(hiveMetricType type, hiveMetricLevel level, u64 *toAdd, bool *sub, hivePerformanceUnit * metric);
void hiveInternalHandleRemoteMetricUpdate(hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub);
hiveMetricLevel hiveInternalUpdatePerformanceMetric(hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub);
hiveMetricLevel hiveInternalUpdatePerformanceCoreMetric(unsigned int core, hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub);
void hiveInternalWriteMetricShotFile(unsigned int threadId, unsigned int nodeId);
void internalPrintTotals(unsigned int nodeId);
void printModelTotalMetrics(hiveMetricLevel level);
void hiveInternalUpdatePacketInfo(u64 bytes);
void hiveInternalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket);
void hiveInternalIntervalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket);
void printInspectorStats(void);
void printInspectorTime(void);
void hiveInternalSetThreadPerformanceMetric(hiveMetricType type, u64 value);
u64 hiveGetInspectorTime(void);
#ifdef INSPECTOR

#define hiveReadInspectorConfigFile(filename) hiveInternalReadInspectorConfigFile(filename)
#define hiveInitIntrospector(config) hiveInternalInitIntrospector(config)
#define hiveStartInspector(startPoint) hiveInternalStartInspector(startPoint)
#define hiveStopInspector() hiveInternalStopInspector()
#define hiveInspecting() hiveInternalInspecting()
#define hiveGetPerformanceMetricTotal(type, level) hiveInternalGetPerformanceMetricTotal(type, level)
#define hiveGetPerformanceMetricRate(type, level, last) hiveInternalGetPerformanceMetricRate(type, level, last)
#define hiveGetPerformanceMetricRateU64(type, level, last) hiveInternalGetPerformanceMetricRateU64(type, level, last)
#define hiveGetPerformanceMetricRateU64Diff(type, level, diff) hiveInternalGetPerformanceMetricRateU64Diff(type, level, diff)
#define hiveGetTotalMetricRateU64(type, level, total, timeStamp) hiveInternalGetTotalMetricRateU64(type, level, total, timeStamp)
#define hiveSingleMetricUpdate(type, level, toAdd, sub, metric) hiveInternalSingleMetricUpdate(type, level, toAdd, sub, metric)
#define hiveUpdatePerformanceMetric(type, level, toAdd, sub) hiveInternalUpdatePerformanceMetric(type, level, toAdd, sub)
#define hiveUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) hiveInternalUpdatePerformanceCoreMetric(core, type, level, toAdd, sub)
#define hiveWriteMetricShotFile(threadId, nodeId) hiveInternalWriteMetricShotFile(threadId, nodeId)
#define hiveHandleRemoteMetricUpdate(type, level, toAdd, sub) hiveInternalHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define hiveIntrospectivePrintTotals(nodeId) internalPrintTotals(nodeId)
#define hiveUpdatePacketInfo(bytes) hiveInternalUpdatePacketInfo(bytes)
#define hivePacketStats(totalBytes, totalPackets, minPacket, maxPacket) hiveInternalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define hiveIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket) hiveInternalIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define hiveSetThreadPerformanceMetric(type, value) hiveInternalSetThreadPerformanceMetric(type, value)

#else

#define hiveReadInspectorConfigFile(filename)
#define hiveInitIntrospector(config)
#define hiveStartInspector(startPoint)
#define hiveStopInspector()
#define hiveInspecting() 0
#define hiveGetPerformanceMetricTotal(type, level) 0
#define hiveGetPerformanceMetricRate(type, level, last) 0
#define hiveGetPerformanceMetricRateU64(type, level, last) 0
#define hiveGetPerformanceMetricRateU64Diff(type, level, diff) 0
#define hiveGetTotalMetricRateU64(type, level, total, timeStamp) 0
#define hiveSingleMetricUpdate(type, level, toAdd, sub, metric, timeStamp) 0
#define hiveUpdatePerformanceMetric(type, level, toAdd, sub) -1
#define hiveUpdatePerformanceCoreMetric(core, type, level, toAdd, sub) -1
#define hiveWriteMetricShotFile(threadId, nodeId)
#define hiveHandleRemoteMetricUpdate(type, level, toAdd, sub)
#define hiveIntrospectivePrintTotals(nodeId)
#define hiveUpdatePacketInfo(bytes)
#define hivePacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define hiveIntervalPacketStats(totalBytes, totalPackets, minPacket, maxPacket)
#define hiveSetThreadPerformanceMetric(type, value)

#endif
#ifdef __cplusplus
}
#endif

#endif /* hiveINTROSPECTION_H */

