#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"
#include "artsAtomics.h"

unsigned int counter = 0;
uint64_t numDummy = 0;
artsGuid_t exitGuid = NULL_GUID;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
}

void syncTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{ 
    PRINTF("Guid: %lu Sync %lu: %lu\n", artsGetCurrentGuid(), paramv[0], depv[0].guid);
}

void exitProgram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    PRINTF("Exit: %lu\n", depv[0].guid);
    artsShutdown();
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    uint64_t dep = paramv[0];
    if(dep)
    {
        dep--;
        artsGuid_t guid = artsEdtCreate(syncTask, artsGetCurrentNode(), 1, &dep, 1);
        artsGuid_t epochGuid = artsInitializeAndStartEpoch(guid, 0);
        PRINTF("Guid: %lu Root: %lu sync: %lu epoch: %lu\n", artsGetCurrentGuid(), dep, guid, epochGuid);
        
        unsigned int numNodes = artsGetTotalNodes();
        for (unsigned int rank = 0; rank < numNodes; rank++)
            artsEdtCreate(rootTask, rank % numNodes, 1, &dep, 0);
        
        for (uint64_t rank = 0; rank < numNodes * numDummy; rank++)
            artsEdtCreate(dummytask, rank % numNodes, 0, NULL, 0);
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (uint64_t) atoi(argv[1]);
    exitGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId)
    {
        if(!workerId)
        {
            PRINTF("Starting\n");
            artsEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, 1);
            artsGuid_t epochGuid = artsInitializeAndStartEpoch(exitGuid, 0);
            artsGuid_t startGuid = artsEdtCreate(rootTask, 0, 1, &numDummy, 0);
        }
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
