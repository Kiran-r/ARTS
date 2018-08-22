#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"
#include "artsAtomics.h"

unsigned int counter = 0;
unsigned int numDummy = 0;
artsGuid_t exitGuid = NULL_GUID;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    artsAtomicAdd(&counter, 1);
}

void exitProgram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    unsigned int numNodes = artsGetTotalNodes();
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int numEdts = depv[i].guid;
        if(numEdts!=numNodes*numDummy+2)
            PRINTF("Error: %u vs %u\n", numEdts, numNodes*numDummy+2);
    }
    PRINTF("Exit %u\n", counter);
    artsShutdown();
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    artsGuid_t guid = artsGetCurrentEpochGuid();
    PRINTF("Starting %lu %u\n", guid, artsGuidGetRank(guid));
    unsigned int numNodes = artsGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes * numDummy; rank++)
        artsEdtCreate(dummytask, rank % numNodes, 0, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (unsigned int) atoi(argv[1]);
    exitGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId)
        artsEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, artsGetTotalNodes());
    
    if (!workerId) 
    {
        artsInitializeAndStartEpoch(exitGuid, nodeId);
        artsGuid_t startGuid = artsEdtCreate(rootTask, nodeId, 0, NULL, 0);
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
