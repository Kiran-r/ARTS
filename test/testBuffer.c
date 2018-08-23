#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

artsGuid_t dbDestGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

void dummy(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t resultGuid = paramv[0];
    unsigned int resultSize = paramv[1];
    unsigned int bufferSize = paramv[2]/sizeof(unsigned int);
    unsigned int * buffer = depv[0].ptr;
    PRINTF("%lu %u %u %p\n", resultGuid, resultSize, bufferSize, buffer);
    unsigned int * sum = artsCalloc(resultSize);
    for(unsigned int i=0; i<bufferSize; i++)
    {
        PRINTF("%u\n", buffer[i]);
        *sum+=buffer[i];
    }
    PRINTF("Sum before: %u\n", *sum);
    artsSetBuffer(resultGuid, sum, resultSize);
}

void startEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t args[3];
    
    unsigned int result = 0;
    unsigned int * dataPtr = &result;
    args[0] = artsAllocateLocalBuffer((void**)&dataPtr, sizeof(unsigned int), 1, NULL_GUID);
    args[1] = sizeof(unsigned int);
    
    unsigned int bufferSize = sizeof(unsigned int)*5;
    unsigned int * data = artsCalloc(bufferSize);
    for(unsigned int i=0; i<5; i++)
        data[i] = i;
    args[2] = bufferSize;
    
    artsActiveMessageWithBuffer(dummy, (artsGetCurrentNode()+1) % artsGetTotalNodes(), 3, args, 0, data, bufferSize);
    
    while(!result)
    {
        artsYield();
        PRINTF("Did a yield\n");
    }
    
    PRINTF("Sum: %u\n", result);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    PRINTF("Starting\n");
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId && !workerId)
    {
        artsEdtCreate(startEdt, 0, 0, NULL, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}