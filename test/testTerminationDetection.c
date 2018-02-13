#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "hiveAtomics.h"

unsigned int counter = 0;
unsigned int numDummy = 0;
hiveGuid_t exitGuid = NULL_GUID;

hiveGuid_t dummytask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    hiveAtomicAdd(&counter, 1);
}

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    unsigned int numNodes = hiveGetTotalNodes();
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int numEdts = depv[i].guid;
        if(numEdts!=numNodes*numDummy+1)
            PRINTF("Error: %u vs %u\n", numEdts, numNodes*numDummy+1);
    }
    PRINTF("Exit %u\n", counter);
    hiveShutdown();
}

hiveGuid_t rootTask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    hiveGuid_t guid = hiveGetCurrentEpochGuid();
    PRINTF("Starting %lu %u\n", guid, hiveGuidGetRank(guid));
    unsigned int numNodes = hiveGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes * numDummy; rank++)
        hiveEdtCreate(dummytask, rank % numNodes, 0, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (unsigned int) atoi(argv[1]);
    exitGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId)
        hiveEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, hiveGetTotalNodes());
    
    if (!workerId) 
    {
        hiveGuid_t startGuid = hiveEdtCreate(rootTask, nodeId, 0, NULL, 1);
        hiveInitializeEpoch(startGuid, exitGuid, nodeId);
        hiveSignalEdt(startGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
    }
}

int main(int argc, char** argv) 
{
    hiveRT(argc, argv);
    return 0;
}
