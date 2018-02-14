#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
hiveArrayDb_t * array = NULL;

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u\n", data[j]);
        }
    }
    hiveShutdown();
}

hiveGuid_t epochEnd(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{    
    hiveGatherArrayDb(array, check, 0, 0, NULL, 0);
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = atoi(argv[2]);
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!nodeId && !workerId)
    {
        hiveGuid_t endEpochGuid = hiveEdtCreate(epochEnd, 0, 0, NULL, 1);
        hiveInitializeAndStartEpoch(endEpochGuid, 0);
        
        hiveGuid_t guid = hiveNewArrayDb(&array, sizeof(unsigned int), elementsPerBlock, blocks);
        for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            hivePutInArrayDb(&i, NULL_GUID, 0, array, i);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
