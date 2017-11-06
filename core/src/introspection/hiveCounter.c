#include <unistd.h>
#include "hive.h"
#include "hiveMalloc.h"
#include "hiveCounter.h"
#include "hiveGlobals.h"
#include "hiveUtil.h"
#include "hiveArrayList.h"
#include "hiveAtomics.h"
#include "hiveIntrospection.h"
#include <string.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>

char * counterPrefix;
unsigned int countersOn = 0;
unsigned int counterStartPoint = 0;

void hiveInitCounterList(unsigned int threadId, unsigned int nodeId, char * folder, unsigned int startPoint)
{
    counterPrefix = folder;
    counterStartPoint = startPoint;
    COUNTERNAMES;
    hiveThreadInfo.counterList = hiveNewArrayList(sizeof(hiveCounter), COUNTERARRAYBLOCKSIZE);
    for(int i=FIRSTCOUNTER; i<LASTCOUNTER; i++)
    {   
        hivePushToArrayList(hiveThreadInfo.counterList, hiveCreateCounter(threadId, nodeId, GETCOUNTERNAME(i) ));
    }
    if(counterStartPoint == 1)
        countersOn = 1;
}

void hiveStartCounters(unsigned int startPoint)
{
    if(counterStartPoint == startPoint)
    {
//        PRINTF("TURNING COUNTERS ON");
        countersOn = 1;
    }
}

unsigned int hiveCountersOn()
{
    return countersOn;
}

void hiveEndCounters()
{
    countersOn = 0;
}

hiveCounter * hiveCreateCounter(unsigned int threadId, unsigned int nodeId, const char * counterName)
{
    hiveCounter * counter = (hiveCounter*) hiveMalloc(sizeof(hiveCounter));
    counter->threadId = threadId;
    counter->nodeId = nodeId;
    counter->name = counterName;
    hiveResetCounter(counter);
    return counter;
}

hiveCounter * hiveUserGetCounter(unsigned int index, char * name)
{
    unsigned int currentSize = (unsigned int) hiveLengthArrayList(hiveThreadInfo.counterList);
    for(unsigned int i=currentSize; i<=index; i++)
        hivePushToArrayList(hiveThreadInfo.counterList, hiveCreateCounter(hiveThreadInfo.coreId, hiveGlobalRankId, NULL ));
    hiveCounter * counter = hiveGetCounter(index);
    if(counter->name == NULL)
        counter->name = name;
    return counter;
}

hiveCounter * hiveGetCounter(hiveCounterType counter)
{
    return (hiveCounter*) hiveGetFromArrayList(hiveThreadInfo.counterList, counter);
}

void hiveResetCounter(hiveCounter * counter)
{
    counter->count = 0;
    counter->totalTime = 0;
    counter->startTime = 0;
    counter->endTime = 0;
}

void hiveCounterIncrement(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        counter->count++;
}

void hiveCounterIncrementBy(hiveCounter * counter, uint64_t num)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        counter->count+=num;
}

void hiveCounterTimerStart(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        counter->startTime = COUNTERTIMESTAMP;
}

void hiveCounterTimerEndIncrement(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
    {
        counter->endTime = COUNTERTIMESTAMP;
        counter->totalTime+=(counter->endTime - counter->startTime);
        counter->count++;
    }
}

void hiveCounterTimerEndOverwrite(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
    {
        counter->endTime = COUNTERTIMESTAMP;
        counter->totalTime=(counter->endTime - counter->startTime);
        counter->count++;
    }
}

void hiveCounterAddTime(hiveCounter * counter, uint64_t time)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
    {
        counter->totalTime+=time;
        counter->count++;
    }
}

void hiveCounterAddEndTime(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
    {
        if(counter->startTime && counter->endTime)
        {
            counter->totalTime+=counter->endTime - counter->startTime;
            counter->count++;
            counter->startTime = 0;
            counter->endTime = 0;
        }
    }
}

void hiveCounterNonEmtpy(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
    {
        if(!counter->startTime)
        {
            counter->startTime = counter->endTime;
            counter->endTime = COUNTERTIMESTAMP;
        }
    }
}

void hiveCounterSetStartTime(hiveCounter * counter, uint64_t start)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        counter->startTime = start;
}

void hiveCounterSetEndTime(hiveCounter * counter, uint64_t end)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        counter->endTime = end;
}

uint64_t hiveCounterGetStartTime(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        return counter->startTime;
    else
        return 0;
}

uint64_t hiveCounterGetEndTime(hiveCounter * counter)
{
    if(counter && countersOn && hiveThreadInfo.localCounting)
        return counter->endTime;
    else
        return 0;
}

void hiveCounterPrint(hiveCounter * counter, FILE * stream)
{
    fprintf(stream, "%s %u %u %" PRIu64 " %" PRIu64 "\n", counter->name, counter->nodeId, counter->threadId, counter->count, counter->totalTime);
}

void hiveWriteCountersToFile(unsigned int threadId, unsigned int nodeId)
{
    if(countersOn)
    {
        char * filename;
        if(counterPrefix)
        {
            struct stat st = {0};
            if (stat(counterPrefix, &st) == -1)
                mkdir(counterPrefix, 0755);
            
            unsigned int stringSize = strlen(counterPrefix) + COUNTERPREFIXSIZE;
            filename = hiveMalloc(sizeof(char)*stringSize);
            sprintf(filename,"%s/%s_%u_%u.ct", counterPrefix, "counter", nodeId, threadId);
        }
        else
        {
            filename = hiveMalloc(sizeof(char)*COUNTERPREFIXSIZE);
            sprintf(filename,"%s_%u_%u.ct", "counter", nodeId, threadId);
        }

        FILE * fp = fopen(filename,"w");
        if(fp)
        {
            uint64_t i;
            uint64_t length = hiveLengthArrayList(hiveThreadInfo.counterList);
            for(i=0; i<length; i++)
            {
                hiveCounter * counter = hiveGetFromArrayList(hiveThreadInfo.counterList, i);
                if(counter->name)
                    hiveCounterPrint(counter, fp);
            }
    //        hiveDeleteArrayList(hiveThreadInfo.counterList);
        }
        else
            printf("Failed to open %s\n", filename);
    }
}
