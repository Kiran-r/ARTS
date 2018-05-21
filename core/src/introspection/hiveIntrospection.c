#include "hive.h"
#include "hiveConfig.h"
#include "hiveGlobals.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveArrayList.h"
#include "hiveRemoteFunctions.h"
#include "hiveIntrospection.h"
#include "hiveDebug.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "stdint.h"
#include "inttypes.h"
#define DPRINTF( ... )
#define NANOSECS 1000000000

hiveMETRICNAME;
u64 ** countWindow;
u64 ** timeWindow;
u64 ** maxTotal;

char * printTotalsToFile = NULL;
volatile unsigned int inspectorOn = 0;
hiveInspector * inspector = NULL;
hiveInspectorStats * stats = NULL;
hiveInspectorShots * inspectorShots = NULL;
hivePacketInspector * packetInspector = NULL;

u64 localTimeStamp(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
//    clock_gettime(CLOCK_BOOTTIME, &res);
    u64 timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}

u64 globalTimeStamp(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    u64 timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}

u64 hiveGetInspectorTime(void)
{
    return inspector->startTimeStamp;
}

bool hiveInternalInspecting(void)
{
    return (inspectorOn);
}

void hiveInternalStartInspector(unsigned int startPoint)
{
    if(inspector && inspector->startPoint == startPoint)
    {
        inspectorOn = 1;
        inspector->startTimeStamp = globalTimeStamp();
//        PRINTF("TURNING INSPECTION ON Folder: %s %ld\n", printTotalsToFile, inspector->startTimeStamp);
    }
}

void hiveInternalStopInspector(void)
{
    if(inspector)
    {
        inspectorOn = 0;
        inspector->endTimeStamp = globalTimeStamp();
    }
}

void printMetrics(void)
{
    for(unsigned int i=0; i<hiveLastMetricType; i++)
    {
        PRINTF("%35s %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",
                hiveMetricName[i], 
                countWindow[i][0], countWindow[i][1], countWindow[i][2], 
                timeWindow[i][0], timeWindow[i][1], timeWindow[i][2],
                maxTotal[i][0], maxTotal[i][1], maxTotal[i][2]);
    }
}

void hiveInternalInitIntrospector(struct hiveConfig * config)
{
    char * inspFileName = config->introspectiveConf;
    char * inspOutputPrefix = config->introspectiveFolder;
    unsigned int traceLevel = config->introspectiveTraceLevel;
    unsigned int startPoint = config->introspectiveStartPoint;
    
    if(inspFileName)
    {
        DPRINTF("countWindow %u\n", sizeof(u64*) * hiveLastMetricType);
        countWindow = hiveMalloc(sizeof(u64*) * hiveLastMetricType);
        DPRINTF("timeWindow %u\n", sizeof(u64*) * hiveLastMetricType);
        timeWindow = hiveMalloc(sizeof(u64*) * hiveLastMetricType);
        DPRINTF("maxTotal %u\n", sizeof(u64*) * hiveLastMetricType);
        maxTotal = hiveMalloc(sizeof(u64*) * hiveLastMetricType);
        
        for(unsigned int i=0; i<hiveLastMetricType; i++)
        {
            DPRINTF("countWindow[%u] %u\n",i, sizeof(u64) * hiveMETRICLEVELS);
            countWindow[i] = hiveMalloc(sizeof(u64) * hiveMETRICLEVELS);
            DPRINTF("timeWindow[%u] %u\n",i, sizeof(u64) * hiveMETRICLEVELS);
            timeWindow[i] = hiveMalloc(sizeof(u64) * hiveMETRICLEVELS);
            DPRINTF("maxTotal[%u] %u\n",i, sizeof(u64) * hiveMETRICLEVELS);
            maxTotal[i] = hiveMalloc(sizeof(u64) * hiveMETRICLEVELS);
            for(unsigned int j=0; j<hiveMETRICLEVELS; j++)
            {
                countWindow[i][j] = -1;
                timeWindow[i][j] = -1;
                maxTotal[i][j] = -1;
            }
        }

        hiveInternalReadInspectorConfigFile(inspFileName);
        if(!hiveGlobalRankId)
            printMetrics();
        DPRINTF("inspector %u\n", sizeof(hiveInspector));
        inspector = hiveCalloc(sizeof(hiveInspector));
        inspector->startPoint = startPoint;
        DPRINTF("inspector->coreMetric %u\n", sizeof(hivePerformanceUnit) * hiveLastMetricType * hiveNodeInfo.totalThreadCount);
        inspector->coreMetric = hiveCalloc(sizeof(hivePerformanceUnit) * hiveLastMetricType * hiveNodeInfo.totalThreadCount);
        for(unsigned int i=0; i<hiveNodeInfo.totalThreadCount; i++)
        {
            for(unsigned int j=0; j<hiveLastMetricType; j++)
            {
                inspector->coreMetric[i*hiveLastMetricType + j].maxTotal = maxTotal[j][0];
                inspector->coreMetric[i*hiveLastMetricType + j].timeMethod = localTimeStamp;
            }
        }
        
        inspector->nodeMetric = hiveCalloc(sizeof(hivePerformanceUnit) * hiveLastMetricType);
        for(unsigned int j=0; j<hiveLastMetricType; j++)
        {
            inspector->nodeMetric[j].maxTotal = maxTotal[j][1];
            inspector->nodeMetric[j].timeMethod = globalTimeStamp;
        }
        
        inspector->systemMetric = hiveCalloc(sizeof(hivePerformanceUnit) * hiveLastMetricType);
        for(unsigned int j=0; j<hiveLastMetricType; j++)
        {
            inspector->systemMetric[j].maxTotal = maxTotal[j][2];
            inspector->systemMetric[j].timeMethod = globalTimeStamp;
        }
        
        DPRINTF("stats %u\n", sizeof(hiveInspectorStats));
        stats = hiveCalloc(sizeof(hiveInspectorStats));
        DPRINTF("packetInspector %u\n", sizeof(hivePacketInspector));
        packetInspector = hiveCalloc(sizeof(hivePacketInspector));
        packetInspector->minPacket = (u64) -1;
        packetInspector->maxPacket = 0;
        packetInspector->intervalMin = (u64) -1;
        packetInspector->intervalMax = 0;
        
        if(inspOutputPrefix && traceLevel < hiveMETRICLEVELS)
        {
            if(traceLevel<=hiveSystem)
            {
                inspectorShots = hiveMalloc(sizeof(hiveInspectorShots));
                DPRINTF("inspectorShots->coreMetric\n");
                inspectorShots->coreMetric = hiveCalloc(sizeof(hiveArrayList*) * hiveLastMetricType * hiveNodeInfo.totalThreadCount);
                for(unsigned int i = 0; i < hiveLastMetricType * hiveNodeInfo.totalThreadCount; i++)
                    inspectorShots->coreMetric[i] = hiveNewArrayList(sizeof(hiveMetricShot), 1024);
                DPRINTF("inspectorShots->nodeMetric\n");
                inspectorShots->nodeMetric = hiveCalloc(sizeof(hiveArrayList*) * hiveLastMetricType);
                for(unsigned int i = 0; i < hiveLastMetricType; i++)
                    inspectorShots->nodeMetric[i] = hiveNewArrayList(sizeof(hiveMetricShot), 1024);
                DPRINTF("inspectorShots->systemMetric\n");
                inspectorShots->systemMetric = hiveCalloc(sizeof(hiveArrayList*) * hiveLastMetricType);
                for(unsigned int i = 0; i < hiveLastMetricType; i++)
                    inspectorShots->systemMetric[i] = hiveNewArrayList(sizeof(hiveMetricShot), 1024);
                DPRINTF("inspectorShots->nodeLock %u\n", sizeof(unsigned int) * hiveLastMetricType);
                inspectorShots->nodeLock = hiveCalloc(sizeof(unsigned int) * hiveLastMetricType);
                DPRINTF("inspectorShots->systemLock %u\n", sizeof(unsigned int) * hiveLastMetricType);
                inspectorShots->systemLock = hiveCalloc(sizeof(unsigned int) * hiveLastMetricType);
                inspectorShots->prefix = inspOutputPrefix;
                inspectorShots->traceLevel = (hiveMetricLevel) traceLevel;
            }
        }
        if(inspOutputPrefix && traceLevel <= hiveMETRICLEVELS)
            printTotalsToFile = inspOutputPrefix;
    }
}

bool metricTryLock(hiveMetricLevel level, hivePerformanceUnit * metric)
{
    if(level == hiveThread)
        return true;
    
    unsigned int local;
    while(1)
    {
        local = hiveAtomicCswap(&metric->lock, 0U, 1U);
        if(local != 2U)
            break;
    }
    return (local == 0U);
}

void metricLock(hiveMetricLevel level, hivePerformanceUnit * metric)
{
    if(level == hiveThread)
        return;
    while(!hiveAtomicCswap(&metric->lock, 0U, 1U));
}

void metricUnlock(hivePerformanceUnit * metric)
{
    metric->lock=0U;
}

hivePerformanceUnit * getMetric(hiveMetricType type, hiveMetricLevel level)
{
    hivePerformanceUnit * metric = NULL;
    if(inspector)
    {
        switch(level)
        {
            case hiveThread:
                metric = &inspector->coreMetric[hiveThreadInfo.threadId*hiveLastMetricType + type];
                break;
            case hiveNode:
                metric = &inspector->nodeMetric[type];
                break;
            case hiveSystem:
                metric = &inspector->systemMetric[type];
                break;
            default:
                metric = NULL;
                break;
        }
    }
    return metric;
}

u64 hiveInternalGetPerformanceMetricTotal(hiveMetricType type, hiveMetricLevel level)
{
    
    hivePerformanceUnit * metric = getMetric(type, level);
    return (metric) ? metric->totalCount : 0;
}

double hiveInternalGetPerformanceMetricRate(hiveMetricType type, hiveMetricLevel level, bool last)
{
    hivePerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        u64 localWindowTimeStamp;
        u64 localWindowCountStamp;
        u64 localCurrentCountStamp;
        u64 localCurrentTimeStamp;
        
        metricLock(level, metric);
        if(last)
        {
            localWindowTimeStamp = metric->lastWindowTimeStamp;
            localWindowCountStamp = metric->lastWindowCountStamp;
            localCurrentCountStamp = metric->windowCountStamp;
            localCurrentTimeStamp = metric->windowTimeStamp;
            metricUnlock(metric);
        }
        else
        {
            localWindowTimeStamp = metric->windowTimeStamp;
            localWindowCountStamp = metric->windowCountStamp;
            metricUnlock(metric);
            localCurrentCountStamp = metric->totalCount;
            localCurrentTimeStamp = metric->timeMethod();
        }       

        if(localCurrentCountStamp && localCurrentTimeStamp)
            return (localCurrentCountStamp - localWindowCountStamp) / ((localCurrentTimeStamp - localWindowTimeStamp) / 1E9);
    }
    return 0;
}

u64 hiveInternalGetPerformanceMetricRateU64(hiveMetricType type, hiveMetricLevel level, bool last)
{
    hivePerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        u64 localWindowTimeStamp;
        u64 localWindowCountStamp;
        u64 localCurrentCountStamp;
        u64 localCurrentTimeStamp;
        
        metricLock(level, metric);
        if(last)
        {
            localWindowTimeStamp = metric->lastWindowTimeStamp;
            localWindowCountStamp = metric->lastWindowCountStamp;
            localCurrentCountStamp = metric->windowCountStamp;
            localCurrentTimeStamp = metric->windowTimeStamp;
            metricUnlock(metric);
        }
        else
        {
            localWindowTimeStamp = metric->windowTimeStamp;
            localWindowCountStamp = metric->windowCountStamp;
            metricUnlock(metric);
            localCurrentCountStamp = metric->totalCount;
            localCurrentTimeStamp = metric->timeMethod();
        }       
       
        if(localCurrentCountStamp && localCurrentTimeStamp && localCurrentCountStamp > localWindowCountStamp)
        {
//            PRINTF("%lu / %lu = %lu\n", (localCurrentTimeStamp - localWindowTimeStamp), (localCurrentCountStamp - localWindowCountStamp), ((localCurrentTimeStamp - localWindowTimeStamp) / (localCurrentCountStamp - localWindowCountStamp)));
            return (localCurrentTimeStamp - localWindowTimeStamp) / (localCurrentCountStamp - localWindowCountStamp);
        }
    }
    return 0;
}

u64 hiveInternalGetPerformanceMetricRateU64Diff(hiveMetricType type, hiveMetricLevel level, u64 * total)
{
    hivePerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        metricLock(level, metric);
        u64 localWindowTimeStamp =  metric->windowTimeStamp;
        u64 localWindowCountStamp =  metric->windowCountStamp;
        u64 lastWindowTimeStamp =  metric->lastWindowTimeStamp;
        u64 lastWindowCountStamp =  metric->lastWindowCountStamp;
        metricUnlock(metric);
        
        u64 localCurrentCountStamp = metric->totalCount;
        u64 localCurrentTimeStamp = metric->timeMethod();
        *total = localCurrentCountStamp;
        if(localCurrentCountStamp)
        {
            u64 diff = localCurrentCountStamp - localWindowCountStamp;
            if(diff && localWindowTimeStamp)
            {
                return (localCurrentTimeStamp - localWindowTimeStamp) / diff;
            }
            else 
            {
                diff = localWindowCountStamp - lastWindowCountStamp;
                if(diff && localWindowCountStamp && lastWindowTimeStamp)
                {
                    return (localWindowCountStamp - lastWindowTimeStamp) / diff;
                }
            }
        }
        
    }
    return 0;
}

u64 hiveInternalGetTotalMetricRateU64(hiveMetricType type, hiveMetricLevel level, u64 * total, u64 * timeStamp)
{
    hivePerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        u64 localCurrentCountStamp = *total = metric->totalCount;
        u64 localCurrentTimeStamp = metric->timeMethod();
        *timeStamp = localCurrentTimeStamp;
        u64 startTime = metric->firstTimeStamp;
        if(startTime && localCurrentCountStamp)
        {
//            if(!hiveGlobalRankId && !hiveThreadInfo.threadId)
//                PRINTF("TIME: %lu COUNT: %lu RATE: %lu\n", (localCurrentTimeStamp - startTime), localCurrentCountStamp, (localCurrentTimeStamp - startTime) / localCurrentCountStamp);
            return (localCurrentTimeStamp - startTime) / localCurrentCountStamp;
        }
    }
    return 0;
}

void hiveInternalHandleRemoteMetricUpdate(hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub)
{
    hivePerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        metricLock(level, metric);
        if(sub)
        {
            metric->windowCountStamp -= toAdd;
            metric->totalCount -= toAdd;
        }
        else
        {
            metric->windowCountStamp += toAdd;
            metric->totalCount += toAdd;
        }
        metricUnlock(metric);
        hiveAtomicAddU64(&stats->remoteUpdates, 1);
    }
}

void internalUpdateMax(hiveMetricLevel level, hivePerformanceUnit * metric, u64 total)
{
    u64 entry = metric->maxTotal;
    u64 localMax = metric->maxTotal;
    if(localMax>total)
        return;
    if(level==hiveThread)
        metric->maxTotal = total;
    else
    {
        while(localMax < total)
        {
            localMax = hiveAtomicCswapU64(&metric->maxTotal, localMax, total);
        }
    }
}

u64 internalObserveMax(hiveMetricLevel level, hivePerformanceUnit * metric)
{
    u64 max = -1;
    if(metric->maxTotal!=-1)
    {
        if(level==hiveThread)
        {
            max = metric->maxTotal;
            metric->maxTotal = metric->totalCount;
        }
        else
        {
            max = hiveAtomicSwapU64(&metric->maxTotal, metric->totalCount);
        }
    }
    return max;
}

bool hiveInternalSingleMetricUpdate(hiveMetricType type, hiveMetricLevel level, u64 *toAdd, bool *sub, hivePerformanceUnit * metric)
{
    if(!countWindow[type][level] || !timeWindow[type][level])
        return true;
    
    if(countWindow[type][level] == -1 && timeWindow[type][level] ==-1)
        return false;
    
    u64 totalStamp;
    if(*toAdd)
    {
        if(*sub)
        {
            //Subtraction assumes a zero total sum at the level it is evaluating (you can skip a level by setting a window to 0)
            //It is unclear what a negative result means...
            if(metric->totalCount < *toAdd)
            {
                PRINTF("Potential Inspection Underflow Detected! Level: %s Type: %s\n", level, hiveMetricName[type]);
                hiveDebugPrintStack();
            }
            totalStamp = (level==hiveThread) ? metric->totalCount-=*toAdd : hiveAtomicSubU64(&metric->totalCount, *toAdd);
    //        totalStamp = hiveAtomicSubU64(&metric->totalCount, *toAdd);
        }
        else
        {
            totalStamp = (level==hiveThread) ? metric->totalCount+=*toAdd : hiveAtomicAddU64(&metric->totalCount, *toAdd);
    //        totalStamp = hiveAtomicAddU64(&metric->totalCount, *toAdd);
        }
        internalUpdateMax(level, metric, totalStamp);
    }
    //Read local values to see if we need to update
    u64 localWindowTimeStamp = metric->windowTimeStamp;
    u64 localWindowCountStamp = metric->windowCountStamp;
    
    u64 timeStamp = metric->timeMethod();
    //Check if it is the first timeStamp
    if(!localWindowTimeStamp)
    {
        if(!hiveAtomicCswapU64(&metric->windowTimeStamp, 0, timeStamp))
            metric->firstTimeStamp = metric->windowTimeStamp;
        return false;
    } 
    //Compute the difference in time and counts
    u64 elapsed = (timeStamp > localWindowTimeStamp) ? timeStamp - localWindowTimeStamp : 0;
    u64 last = (totalStamp > localWindowCountStamp) ? totalStamp - localWindowCountStamp : localWindowCountStamp - totalStamp;
        
    if(last >= countWindow[type][level] || elapsed >= timeWindow[type][level])
    {
        if(!metricTryLock(level, metric))
            return false;
        //Check and see if someone else already updated...
        if(localWindowTimeStamp!=metric->windowTimeStamp)
        {
            metricUnlock(metric);
            return false;
        }
        DPRINTF("Check metric %d %d %" PRIu64 " %" PRIu64 " vs %" PRIu64 " %" PRIu64 "\n", level, type, last, elapsed, countWindow[type][level], timeWindow[type][level]);
        DPRINTF("Updating metric %d %d %" PRIu64 " %" PRIu64 "\n", level, type, metric->windowCountStamp, metric->windowTimeStamp);
        //temp store the old
        metric->lastWindowTimeStamp = metric->windowTimeStamp;
        metric->lastWindowCountStamp = metric->windowCountStamp;
        metric->lastWindowMaxTotal = metric->windowMaxTotal;
        //updated to the latest
        metric->windowCountStamp = metric->totalCount;
        metric->windowMaxTotal = internalObserveMax(level, metric);
        metric->windowTimeStamp = metric->timeMethod(); //timeStamp;
        //determine the waterfall
        if(metric->windowCountStamp > metric->lastWindowCountStamp)
        {
            *toAdd = metric->windowCountStamp - metric->lastWindowCountStamp;
            *sub = false;
        }
        else
        {
            *toAdd = metric->lastWindowCountStamp - metric->windowCountStamp;
            *sub = true;
        }
        metricUnlock(metric);
        return true;
    }
    return false;
}

void takeRateShot(hiveMetricType type, hiveMetricLevel level, bool last)
{
    if(inspectorShots && level >= inspectorShots->traceLevel)
    {
        if(!countWindow[type][level] || !timeWindow[type][level])
            return;
        DPRINTF("TRACING LEVEL %d\n", level);
        
        int traceOn = hiveThreadInfo.mallocTrace;
        hiveThreadInfo.mallocTrace = 0;
        hivePerformanceUnit * metric = metric = getMetric(type, level);   
        if(metric)
        {
            hiveArrayList * list = NULL;
            unsigned int * lock = NULL;
            switch(level)
            {
                case hiveThread:
                    list = inspectorShots->coreMetric[hiveThreadInfo.threadId*hiveLastMetricType + type];
                    lock = NULL;
                    break;

                case hiveNode:
                    list = inspectorShots->nodeMetric[type];
                    lock = &inspectorShots->nodeLock[type];
                    break;

                case hiveSystem:
                    list = inspectorShots->systemMetric[type];
                    lock = &inspectorShots->systemLock[type];
                    break;

                default:
                    list = NULL;
                    lock = NULL;
                    break;
            }

            if(list)
            {
                if(lock) 
                {
                    unsigned int local;
                    while(1)
                    {
                        local = hiveAtomicCswap(lock, 0U, 2U );
                        if(local == 2U)
                        {
                            hiveMEMTRACEON;
                            return;
                        }
                        if(!local)
                            break;
                    }
                }
                hiveMetricShot shot;
                if(last)
                {
                    metricLock(level, metric);
                    shot.maxTotal = metric->windowMaxTotal;
                    shot.windowTimeStamp = metric->lastWindowTimeStamp;
                    shot.windowCountStamp = metric->lastWindowCountStamp;
                    shot.currentTimeStamp = metric->windowTimeStamp;
                    shot.currentCountStamp = metric->windowCountStamp;
                    metricUnlock(metric);
                }
                else
                {
                    metricLock(level, metric);
                    shot.windowTimeStamp = metric->windowTimeStamp;
                    shot.windowCountStamp = metric->windowCountStamp;
                    metricUnlock(metric);
                    
                    shot.maxTotal = metric->maxTotal;
                    shot.currentCountStamp = metric->totalCount;
                    shot.currentTimeStamp = metric->timeMethod();
                }

//                if(shot.windowCountStamp && shot.windowTimeStamp)
                hiveThreadInfo.mallocTrace = 0;
                hivePushToArrayList(list, &shot);
                hiveThreadInfo.mallocTrace = 1;
                
                if(lock) 
                    *lock = 0U;
            }
        }
        hiveThreadInfo.mallocTrace = traceOn;
    }
}

hiveMetricLevel hiveInternalUpdatePerformanceCoreMetric(unsigned int core, hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub)
{
    if(type <= hiveFirstMetricType || type >= hiveLastMetricType)
    {
        PRINTF("Wrong Introspection Type %d\n", type);
        hiveDebugGenerateSegFault();
    }
    
    hiveMetricLevel updatedLevel = hiveNoLevel;
    if(inspectorOn)
    {
        switch(level)
        {
            case hiveThread:
                DPRINTF("Thread updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, hiveMetricName[type]);
                if(!hiveInternalSingleMetricUpdate(type, hiveThread, &toAdd, &sub, &inspector->coreMetric[core*hiveLastMetricType + type]))
                    break;
                takeRateShot(type, hiveThread, true);
                updatedLevel = hiveThread;

            case hiveNode:
                DPRINTF("Node   updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, hiveMetricName[type]);
                if(!hiveInternalSingleMetricUpdate(type, hiveNode, &toAdd, &sub, &inspector->nodeMetric[type]))
                    break;
                hiveAtomicAddU64(&stats->nodeUpdates, 1);
                takeRateShot(type, hiveNode, true);
                updatedLevel = hiveNode;

            case hiveSystem:
                DPRINTF("System updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, hiveMetricName[type]);
                if(hiveInternalSingleMetricUpdate(type, hiveSystem, &toAdd, &sub, &inspector->systemMetric[type]))
                {
                    u64 timeToSend = inspector->systemMetric[type].timeMethod();
                    int traceOn = hiveThreadInfo.mallocTrace;
                    hiveThreadInfo.mallocTrace = 0;
                    for(unsigned int i=0; i<hiveGlobalRankCount; i++)
                        if(i!=hiveGlobalRankId)
                            hiveRemoteMetricUpdate(i, type, level, timeToSend, toAdd, sub);
                    hiveThreadInfo.mallocTrace = traceOn;
                    hiveAtomicAddU64(&stats->systemUpdates, 1);
                    if(hiveGlobalRankCount>1)
                        hiveAtomicAddU64(&stats->systemMessages, hiveGlobalRankCount-1);
                    takeRateShot(type, hiveSystem, true);
                    updatedLevel = hiveSystem;
                }
            default:
                break;
        }
    }
    return updatedLevel;
}

void hiveInternalSetThreadPerformanceMetric(hiveMetricType type, u64 value)
{   
    
    if(countWindow[type][hiveThread] == -1 && timeWindow[type][hiveThread] ==-1)
        return;
    
    hivePerformanceUnit * metric = getMetric(type, hiveThread);
    if(metric)
    {
        bool shot = true;
        
        metric->lastWindowCountStamp = metric->windowCountStamp;
        metric->lastWindowTimeStamp = metric->windowTimeStamp;
        metric->lastWindowMaxTotal = metric->maxTotal;
        
        u64 localTime = metric->timeMethod();
        if(!metric->firstTimeStamp)
        {
            shot = false;
            metric->firstTimeStamp = localTime;
        }
        
        metric->totalCount = value;
        metric->windowCountStamp = value;
        metric->windowTimeStamp = localTime;
        if(metric->maxTotal < value)
            metric->maxTotal = value;
        if(shot) // && (elapsed >= timeWindow[type][hiveThread] || last >= countWindow[type][hiveThread]))
        {
//            PRINTF("TAKING SHOT %s %lu\n", hiveMetricName[type], value);
            takeRateShot(type, hiveThread, true); 
        }
    }   
}

hiveMetricLevel hiveInternalUpdatePerformanceMetric(hiveMetricType type, hiveMetricLevel level, u64 toAdd, bool sub)
{
    return hiveInternalUpdatePerformanceCoreMetric(hiveThreadInfo.threadId, type, level, toAdd, sub);
}

void metricPrint(hiveMetricType type, hiveMetricLevel level, hiveMetricShot * shot, FILE * stream)
{
    fprintf(stream, "%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", type, level, shot->windowCountStamp, shot->windowTimeStamp, shot->currentCountStamp, shot->currentTimeStamp, shot->maxTotal);
}

void internalMetricWriteToFile(hiveMetricType type, hiveMetricLevel level, char * filename, hiveArrayList * list)
{
    if(!hiveLengthArrayList(list))
    {
        remove(filename);
        return;
    }
    
    FILE * fp = fopen(filename,"w");
    if(fp)
    {
        DPRINTF("FILE: %s\n", filename);
        u64 last = 0;        
        int traceOn = hiveThreadInfo.mallocTrace;
        hiveThreadInfo.mallocTrace = 0;
        hiveArrayListIterator * iter = hiveNewArrayListIterator(list);
        hiveMetricShot * shot;
        while(hiveArrayListHasNext(iter))
        {
            shot = hiveArrayListNext(iter);
            metricPrint(type, level, shot, fp);
            if(shot->currentTimeStamp < last)
            {
                PRINTF("Out of order snap shot: %s %" PRIu64 "\n", filename, last);
                last = shot->currentTimeStamp;
            }
        }
        hiveDeleteArrayListIterator(iter);
        if(last < inspector->endTimeStamp)
        {
            shot->windowCountStamp = shot->currentCountStamp;
            shot->windowTimeStamp = shot->currentTimeStamp;
            shot->currentTimeStamp = inspector->endTimeStamp;
            metricPrint(type, level, shot, fp);
        }
        hiveThreadInfo.mallocTrace = traceOn;
        fclose(fp);
    }
    else
        PRINTF("Couldn't open %s\n", filename);
}

void hiveInternalWriteMetricShotFile(unsigned int threadId, unsigned int nodeId)
{
    if(inspectorShots)
    {
        struct stat st = {0};
        if (stat(inspectorShots->prefix, &st) == -1)
            mkdir(inspectorShots->prefix, 0755);
        
        hiveArrayList * list;
        char filename[1024];
        
        switch(inspectorShots->traceLevel)
        {
            case hiveThread:
                for(unsigned int i=0; i<hiveLastMetricType; i++)
                {
                    list = inspectorShots->coreMetric[threadId*hiveLastMetricType + i];
                    sprintf(filename,"%s/%s_%s_%u_%u.ct", inspectorShots->prefix, "threadMetric", hiveMetricName[i], nodeId, threadId);
                    internalMetricWriteToFile(i, hiveThread, filename, list);
                }
                
            case hiveNode:
                if(!threadId)
                {
                    for(unsigned int i=0; i<hiveLastMetricType; i++)
                    {
                        list = inspectorShots->nodeMetric[i];
                        sprintf(filename,"%s/%s_%s_%u.ct", inspectorShots->prefix, "nodeMetric", hiveMetricName[i], nodeId);
                        internalMetricWriteToFile(i, hiveNode, filename, list);
                    }
                }
                
            case hiveSystem:
                if(!threadId)// && !nodeId)
                {
                    for(unsigned int i=0; i<hiveLastMetricType; i++)
                    {
                        list = inspectorShots->systemMetric[i];
                        sprintf(filename,"%s/%s_%s_%u.ct", inspectorShots->prefix, "systemMetric", hiveMetricName[i], nodeId);
                        internalMetricWriteToFile(i, hiveSystem, filename, list);
                    }
                }
                
            default:
                break;
        }        
    }
}

void hiveInternalReadInspectorConfigFile(char * filename)
{
    char * line = NULL;
    size_t length = 0;
    FILE * fp = fopen(filename,"r");
    if(!fp)
        return;
    
    char temp[hiveMAXMETRICNAME];
    
    while (getline(&line, &length, fp) != -1) 
    {
        DPRINTF("%s", line);
        if(line[0]!='#')
        {
            int paramRead = 0;
            sscanf(line, "%s", temp);
            size_t offset = strlen(temp);
            unsigned int metricIndex = -1;
            for(unsigned int i=0; i<hiveLastMetricType; i++)
            {
                if(!strcmp(temp, hiveMetricName[i]))
                {
                    metricIndex = i;
                    break;
                }
            }

            if(metricIndex >= 0 && metricIndex < hiveLastMetricType)
            {
                for(unsigned int i=0; i<hiveMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &countWindow[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, countWindow[metricIndex][i]);
                }

                for(unsigned int i=0; i<hiveMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &timeWindow[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, timeWindow[metricIndex][i]);
                }
                
                for(unsigned int i=0; i<hiveMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &maxTotal[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, maxTotal[metricIndex][i]);
                }
            }
            
            if(metricIndex < 0 || metricIndex >= hiveLastMetricType || paramRead < hiveMETRICLEVELS * 2)
            {
                PRINTF("FAILED to init metric %s\n", temp);
            }
        }
    }
    fclose(fp);
    
    if (line)
        free(line);
}

void internalPrintTotals(unsigned int nodeId)
{
    if(printTotalsToFile)
    {
        struct stat st = {0};
        if (stat(printTotalsToFile, &st) == -1)
            mkdir(printTotalsToFile, 0755);
        
        char filename[1024];
        sprintf(filename,"%s/finalCounts_%u.ct", printTotalsToFile, nodeId);
        FILE * fp = fopen(filename,"w");
        if(fp)
        {
            for(unsigned int i=0; i<hiveLastMetricType; i++)
            {
                fprintf(fp, "%s, System, %" PRIu64 ", %" PRIu64 "\n", hiveMetricName[i], inspector->systemMetric[i].totalCount, inspector->systemMetric[i].maxTotal);  
                fprintf(fp, "%s, Node, %" PRIu64 ", %" PRIu64 "\n", hiveMetricName[i], inspector->nodeMetric[i].totalCount, inspector->nodeMetric[i].maxTotal); 
                for(unsigned int j=0; j<hiveNodeInfo.totalThreadCount; j++)
                    fprintf(fp, "%s, Core_%u, %" PRIu64 ", %" PRIu64 "\n", hiveMetricName[i], j, inspector->coreMetric[j*hiveLastMetricType + i].totalCount, inspector->coreMetric[j*hiveLastMetricType + i].maxTotal);
            }

            u64 counted = 0;
            u64 posNotCounted = 0;
            u64 negNotCounted = 0;
            hivePerformanceUnit * metric;
            for(unsigned int i=0; i<hiveLastMetricType; i++)
            {
                counted = posNotCounted = negNotCounted = 0;
                for(unsigned int j=0; j<hiveNodeInfo.totalThreadCount; j++)
                {   
                    metric = &inspector->coreMetric[j*hiveLastMetricType + i];
                    counted += metric->windowCountStamp;
                    if(metric->totalCount > metric->windowCountStamp)
                        posNotCounted += (metric->totalCount - metric->windowCountStamp);
                    else
                        negNotCounted += (metric->windowCountStamp - metric->totalCount);
                }
                metric = &inspector->nodeMetric[i]; 
                u64 sum = metric->totalCount + posNotCounted - negNotCounted;
                if(metric->totalCount == counted && counted + posNotCounted >= negNotCounted)
                    fprintf(fp, "%s, Match, Sum, %" PRIu64 ", Total Counted, %" PRIu64 ", +Rem, %" PRIu64 ", -Rem, %" PRIu64 "\n", hiveMetricName[i], sum, metric->totalCount, posNotCounted, negNotCounted);
                else
                    fprintf(fp, "%s, Error, Sum, %" PRIu64 ", Total Counted, %" PRIu64 ", +Rem, %" PRIu64 ", -Rem, %" PRIu64 "\n", hiveMetricName[i], sum, metric->totalCount, posNotCounted, negNotCounted);
            }
            fprintf(fp, "Node Updates, %" PRIu64 ", System Updates, %" PRIu64 ", Remote Updates,  %" PRIu64 ", System Messages, %" PRIu64 "\n", stats->nodeUpdates, stats->systemUpdates, stats->remoteUpdates, stats->systemMessages);
        }
        else
            PRINTF("Couldn't open %s\n", filename);
    }
}

void printInspectorTime(void)
{
    printf("Stat 0 Node %u Start %" PRIu64 " End %" PRIu64 "\n", hiveGlobalRankId, inspector->startTimeStamp, inspector->endTimeStamp);
}

void printInspectorStats(void)
{
    printf("Stat 3 Node %u Node_Updates %" PRIu64 " System_Updates %" PRIu64 " Remote_Updates  %" PRIu64 " System_Messages %" PRIu64 "\n", hiveGlobalRankId, stats->nodeUpdates, stats->systemUpdates, stats->remoteUpdates, stats->systemMessages);
}

void printModelTotalMetrics(hiveMetricLevel level)
{
    if(level==hiveNode)
        printf("Stat 1 Node %u edt %" PRIu64 " edt_signal %" PRIu64 " event_signal %" PRIu64 " network_sent %" PRIu64 " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
               hiveGlobalRankId,
               hiveInternalGetPerformanceMetricTotal(hiveEdtThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveEdtSignalThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveEventSignalThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveNetworkSendBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveNetworkRecieveBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveMallocBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveFreeBW, level));
    else if(level==hiveThread)
    {
        PRINTF("Stat 1 Thread %u edt %" PRIu64 " edt_signal %" PRIu64 " event_signal %" PRIu64 " network_sent %" PRIu64 " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
               hiveThreadInfo.threadId,
               hiveInternalGetPerformanceMetricTotal(hiveEdtThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveEdtSignalThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveEventSignalThroughput, level),
               hiveInternalGetPerformanceMetricTotal(hiveNetworkSendBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveNetworkRecieveBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveMallocBW, level),
               hiveInternalGetPerformanceMetricTotal(hiveFreeBW, level));
    }
}

static inline void readerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(1)
    {
        while(*writer);
        hiveAtomicFetchAdd(reader, 1U);
        if(*writer==0)
            break;
        hiveAtomicSub(reader, 1U);
    }
}

static inline void readerUnlock(volatile unsigned int * reader)
{
    hiveAtomicSub(reader, 1U);
}

static inline void writerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(hiveAtomicCswap(writer, 0U, 1U) == 0U);
    while(*reader);
    return;
}

static inline void writeUnlock(volatile unsigned int * writer)
{
    hiveAtomicSwap(writer, 0U);
}


static inline void updatePacketExtreme(u64 val, volatile u64 * old, bool min)
{
    u64 local = *old;
    u64 res;
    if(min)
    {
        while(val < local)
        {
            res = hiveAtomicCswapU64(old, local, val);
            if(res==local)
                break;
            else
                local = res;
        }
    }
    else
    {
        while(val > local)
        {
            res = hiveAtomicCswapU64(old, local, val);
            if(res==local)
                break;
            else
                local = res;
        }
    }
}

void hiveInternalUpdatePacketInfo(u64 bytes)
{
    if(packetInspector)
    {
        readerLock(&packetInspector->reader, &packetInspector->writer);
        hiveAtomicAddU64(&packetInspector->totalBytes, bytes);
        hiveAtomicAddU64(&packetInspector->totalPackets, 1U);
        updatePacketExtreme(bytes, &packetInspector->maxPacket, false);
        updatePacketExtreme(bytes, &packetInspector->minPacket, true);
        readerUnlock(&packetInspector->reader);

        readerLock(&packetInspector->intervalReader, &packetInspector->intervalWriter);
        hiveAtomicAddU64(&packetInspector->intervalBytes, bytes);
        hiveAtomicAddU64(&packetInspector->intervalPackets, 1U);
        updatePacketExtreme(bytes, &packetInspector->intervalMax, false);
        updatePacketExtreme(bytes, &packetInspector->intervalMin, true);
        readerUnlock(&packetInspector->intervalReader);
    }
}

void hiveInternalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket)
{
    if(packetInspector)
    {
        writerLock(&packetInspector->reader, &packetInspector->writer);
        (*totalBytes) = packetInspector->totalBytes;
        (*totalPackets) = packetInspector->totalPackets;
        (*minPacket) = packetInspector->minPacket;
        (*maxPacket) = packetInspector->maxPacket;
        writeUnlock(&packetInspector->writer);
    }
}

void hiveInternalIntervalPacketStats(u64 * totalBytes, u64 * totalPackets, u64 * minPacket, u64 * maxPacket)
{
    if(packetInspector)
    {
        writerLock(&packetInspector->intervalReader, &packetInspector->intervalWriter);
        (*totalBytes) = hiveAtomicSwapU64(&packetInspector->totalBytes, 0);
        (*totalPackets) = hiveAtomicSwapU64(&packetInspector->totalPackets, 0);
        (*minPacket) = hiveAtomicSwapU64(&packetInspector->minPacket, 0);
        (*maxPacket) = hiveAtomicSwapU64(&packetInspector->maxPacket, 0);
        writeUnlock(&packetInspector->intervalWriter);
    }
}
