#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
unsigned int numAdd = 0;
artsArrayDb_t * array = NULL;

artsGuid_t end(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc-1; i++)
    {
        unsigned int data = depv[i].guid;
        PRINTF("i: %u updates: %u\n", i, data);
    }
    artsShutdown();
}

//Created by the epochEnd via gather will signal end
artsGuid_t check(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u j: %u %u\n", i, j, data[j]);
        }
    }
    artsSignalEdtValue(paramv[0], (numAdd+1)*elementsPerBlock*blocks, 0);
}

//This is run at the end of the epoch
artsGuid_t epochEnd(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{    
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
    artsGatherArrayDb(array, check, 0, 1, paramv, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = artsGetTotalNodes();
    numAdd = atoi(argv[2]);
    
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && !nodeId)
    {
        //The end will get all the updates and a signal from the gather
        artsGuid_t endGuid = artsEdtCreate(end, 0, 0, NULL, (numAdd+1)*elementsPerBlock*blocks + 1);

        artsGuid_t endEpochGuid = artsEdtCreate(epochEnd, 0, 1, &endGuid, 1);
        artsInitializeAndStartEpoch(endEpochGuid, 0);

        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elementsPerBlock * blocks);

        for(unsigned int j=0; j<numAdd; j++)
        {
            for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            {
                PRINTF("i: %u Slot: %u edt: %lu\n", i, j*elementsPerBlock*blocks + i, endGuid);
                artsAtomicCompareAndSwapInArrayDb(array, i, j, j+1, endGuid, j*elementsPerBlock*blocks + i);
            }
        }

        for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
        {
            PRINTF("i: %u Slot: %u edt: %lu\n", i, numAdd*elementsPerBlock*blocks + i, endGuid);
            artsAtomicCompareAndSwapInArrayDb(array, i, numAdd+1, 0, endGuid, numAdd*elementsPerBlock*blocks + i);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
