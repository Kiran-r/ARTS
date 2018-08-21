#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbDestGuid = NULL_GUID;
hiveGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

hiveGuid_t dummy(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveGuid_t resultGuid = paramv[0];
    unsigned int resultSize = paramv[1];
    unsigned int bufferSize = paramv[2]/sizeof(unsigned int);
    unsigned int * buffer = depv[0].ptr;
    PRINTF("%lu %u %u %p\n", resultGuid, resultSize, bufferSize, buffer);
    unsigned int * sum = hiveCalloc(resultSize);
    for(unsigned int i=0; i<bufferSize; i++)
    {
        PRINTF("%u\n", buffer[i]);
        *sum+=buffer[i];
    }
    PRINTF("Sum before: %u\n", *sum);
    hiveSetBuffer(resultGuid, sum, resultSize);
}

hiveGuid_t startEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    uint64_t args[3];
    
    unsigned int result = 0;
    unsigned int * dataPtr = &result;
    args[0] = hiveAllocateLocalBuffer((void**)&dataPtr, sizeof(unsigned int), 1, NULL_GUID);
    args[1] = sizeof(unsigned int);
    
    unsigned int bufferSize = sizeof(unsigned int)*5;
    unsigned int * data = hiveCalloc(bufferSize);
    for(unsigned int i=0; i<5; i++)
        data[i] = i;
    args[2] = bufferSize;
    
    hiveActiveMessageWithBuffer(dummy, (hiveGetCurrentNode()+1) % hiveGetTotalNodes(), 3, args, 0, data, bufferSize);
    
    while(!result)
    {
        hiveYield();
        PRINTF("Did a yield\n");
    }
    
    PRINTF("Sum: %u\n", result);
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    PRINTF("Starting\n");
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId && !workerId)
    {
        hiveEdtCreate(startEdt, 0, 0, NULL, 0);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}