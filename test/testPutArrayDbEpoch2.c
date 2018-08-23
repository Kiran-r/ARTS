#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
artsArrayDb_t * array = NULL;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u\n", data[j]);
        }
    }
    artsShutdown();
}

void epochEnd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{    
    artsGatherArrayDb(array, check, 0, 0, NULL, 0);
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = artsGetTotalNodes();
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!nodeId && !workerId)
    {
        artsGuid_t endEpochGuid = artsEdtCreate(epochEnd, 0, 0, NULL, 1);
        artsInitializeAndStartEpoch(endEpochGuid, 0);
        
        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elementsPerBlock * blocks);
        for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            artsPutInArrayDb(&i, NULL_GUID, 0, array, i);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
