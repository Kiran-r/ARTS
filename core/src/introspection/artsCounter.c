#include <sys/stat.h>
#include "artsCounter.h"
#include "artsGlobals.h"
#include "artsArrayList.h"
#include "artsAtomics.h"

char * counterPrefix;
uint64_t countersOn = 0;
unsigned int printCounters = 0;
unsigned int counterStartPoint = 0;

void artsInitCounterList(unsigned int threadId, unsigned int nodeId, char * folder, unsigned int startPoint)
{
    counterPrefix = folder;
    counterStartPoint = startPoint;
    COUNTERNAMES;
    artsThreadInfo.counterList = artsNewArrayList(sizeof(artsCounter), COUNTERARRAYBLOCKSIZE);
    for(int i=FIRSTCOUNTER; i<LASTCOUNTER; i++)
    {   
        artsPushToArrayList(artsThreadInfo.counterList, artsCreateCounter(threadId, nodeId, GETCOUNTERNAME(i) ));
    }
    if(counterStartPoint == 1)
    {
        countersOn = COUNTERTIMESTAMP;
        printCounters = 1;
    }
}

void artsStartCounters(unsigned int startPoint)
{
    if(counterStartPoint == startPoint)
    {
//        PRINTF("TURNING COUNTERS ON %u\n", startPoint);
        uint64_t temp = COUNTERTIMESTAMP;
        if(!artsAtomicCswapU64(&countersOn, 0, temp))
        {
            printCounters = 1;
            ARTSCOUNTERTIMERSTART(edtCounter);
        } 
    }
}

unsigned int artsCountersOn()
{
    return countersOn;
}

void artsEndCounters()
{
    uint64_t temp = countersOn;
    countersOn = 0;
    PRINTF("COUNT TIME: %lu countersOn: %lu\n", COUNTERTIMESTAMP - temp, countersOn);
}

artsCounter * artsCreateCounter(unsigned int threadId, unsigned int nodeId, const char * counterName)
{
    artsCounter * counter = (artsCounter*) artsCalloc(sizeof(artsCounter));
    counter->threadId = threadId;
    counter->nodeId = nodeId;
    counter->name = counterName;
    artsResetCounter(counter);
    return counter;
}

artsCounter * artsUserGetCounter(unsigned int index, char * name)
{
    unsigned int currentSize = (unsigned int) artsLengthArrayList(artsThreadInfo.counterList);
    for(unsigned int i=currentSize; i<=index; i++)
        artsPushToArrayList(artsThreadInfo.counterList, artsCreateCounter(artsThreadInfo.coreId, artsGlobalRankId, NULL ));
    artsCounter * counter = artsGetCounter(index);
    if(counter->name == NULL)
        counter->name = name;
    return counter;
}

artsCounter * artsGetCounter(artsCounterType counter)
{
    return (artsCounter*) artsGetFromArrayList(artsThreadInfo.counterList, counter);
}

void artsResetCounter(artsCounter * counter)
{
    counter->count = 0;
    counter->totalTime = 0;
    counter->startTime = 0;
    counter->endTime = 0;
}

void artsCounterIncrement(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        counter->count++;
}

void artsCounterIncrementBy(artsCounter * counter, uint64_t num)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        counter->count+=num;
}

void artsCounterTimerStart(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        counter->startTime = COUNTERTIMESTAMP;
}

void artsCounterTimerEndIncrement(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
    {
        counter->endTime = COUNTERTIMESTAMP;
        counter->totalTime+=(counter->endTime - counter->startTime);
        counter->count++;
    }
}

void artsCounterTimerEndIncrementBy(artsCounter * counter, uint64_t num)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
    {
        counter->endTime = COUNTERTIMESTAMP;
        counter->totalTime+=(counter->endTime - counter->startTime);
        counter->count+=num;
    }
}

void artsCounterTimerEndOverwrite(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
    {
        counter->endTime = COUNTERTIMESTAMP;
        counter->totalTime=(counter->endTime - counter->startTime);
        counter->count++;
    }
}

void artsCounterAddTime(artsCounter * counter, uint64_t time)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
    {
        counter->totalTime+=time;
        counter->count++;
    }
}

void artsCounterAddEndTime(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
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

void artsCounterNonEmtpy(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
    {
        if(!counter->startTime)
        {
            counter->startTime = counter->endTime;
            counter->endTime = COUNTERTIMESTAMP;
        }
    }
}

void artsCounterSetStartTime(artsCounter * counter, uint64_t start)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        counter->startTime = start;
}

void artsCounterSetEndTime(artsCounter * counter, uint64_t end)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        counter->endTime = end;
}

uint64_t artsCounterGetStartTime(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        return counter->startTime;
    else
        return 0;
}

uint64_t artsCounterGetEndTime(artsCounter * counter)
{
    if(counter && countersOn && artsThreadInfo.localCounting)
        return counter->endTime;
    else
        return 0;
}

void artsCounterPrint(artsCounter * counter, FILE * stream)
{
    fprintf(stream, "%s %u %u %" PRIu64 " %" PRIu64 "\n", counter->name, counter->nodeId, counter->threadId, counter->count, counter->totalTime);
}

void artsWriteCountersToFile(unsigned int threadId, unsigned int nodeId)
{
    if(printCounters)
    {
        char * filename;
        if(counterPrefix)
        {
            struct stat st = {0};
            if (stat(counterPrefix, &st) == -1)
                mkdir(counterPrefix, 0755);
            
            unsigned int stringSize = strlen(counterPrefix) + COUNTERPREFIXSIZE;
            filename = artsMalloc(sizeof(char)*stringSize);
            sprintf(filename,"%s/%s_%u_%u.ct", counterPrefix, "counter", nodeId, threadId);
        }
        else
        {
            filename = artsMalloc(sizeof(char)*COUNTERPREFIXSIZE);
            sprintf(filename,"%s_%u_%u.ct", "counter", nodeId, threadId);
        }

        FILE * fp = fopen(filename,"w");
        if(fp)
        {
            uint64_t i;
            uint64_t length = artsLengthArrayList(artsThreadInfo.counterList);
            for(i=0; i<length; i++)
            {
                artsCounter * counter = artsGetFromArrayList(artsThreadInfo.counterList, i);
                if(counter->name)
                    artsCounterPrint(counter, fp);
            }
    //        artsDeleteArrayList(artsThreadInfo.counterList);
        }
        else
            printf("Failed to open %s\n", filename);
    }
}
