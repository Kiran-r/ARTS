#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
unsigned int numAdd = 0;
hiveArrayDb_t * array = NULL;

hiveGuid_t end(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc-1; i++)
    {
        unsigned int data = depv[i].guid;
        PRINTF("updates: %u\n", data);
    }
    hiveShutdown();
}

//Created by the epochEnd via gather will signal end
hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u j: %u %u\n", i, j, data[j]);
        }
    }
    hiveSignalEdt(paramv[0], NULL_GUID, numAdd*elementsPerBlock*blocks, DB_MODE_SINGLE_VALUE);
}

//This is run at the end of the epoch
hiveGuid_t epochEnd(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{    
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
    hiveGatherArrayDb(array, check, 0, 1, paramv, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = hiveGetTotalNodes();
    numAdd = atoi(argv[2]);
    
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && !nodeId)
    {
        //The end will get all the updates and a signal from the gather
        hiveGuid_t endGuid = hiveEdtCreate(end, 0, 0, NULL, numAdd*elementsPerBlock*blocks + 1);

        hiveGuid_t endEpochGuid = hiveEdtCreate(epochEnd, 0, 1, &endGuid, 1);
        hiveInitializeAndStartEpoch(endEpochGuid, 0);

        hiveGuid_t guid = hiveNewArrayDb(&array, sizeof(unsigned int), elementsPerBlock * blocks);

        for(unsigned int j=0; j<numAdd; j++)
        {
            for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            {
                PRINTF("i: %u Slot: %u edt: %lu\n", i, j*elementsPerBlock*blocks + i, endGuid);
                hiveAtomicAddInArrayDb(array, i, 1, endGuid, j*elementsPerBlock*blocks + i);
            }
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
