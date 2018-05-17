#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "hiveAtomics.h"

unsigned int counter = 0;
uint64_t numDummy = 0;
hiveGuid_t exitGuid = NULL_GUID;

hiveGuid_t dummytask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
}

hiveGuid_t syncTask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{ 
//    PRINTF("Guid: %lu Sync %lu: %lu\n", hiveGetCurrentGuid(), paramv[0], depv[0].guid);
}

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    PRINTF("Exit: %lu\n", depv[0].guid);
    hiveShutdown();
}

hiveGuid_t rootTask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    uint64_t dep = paramv[0];
    if(dep)
    {
        dep--;
        hiveGuid_t guid = hiveEdtCreate(syncTask, hiveGetCurrentNode(), 1, &dep, 1);
        hiveGuid_t epochGuid = hiveInitializeAndStartEpoch(guid, 0);
//        PRINTF("Guid: %lu Root: %lu sync: %lu epoch: %lu\n", hiveGetCurrentGuid(), dep, guid, epochGuid);
        
        unsigned int numNodes = hiveGetTotalNodes();
        for (unsigned int rank = 0; rank < numNodes; rank++)
            hiveEdtCreate(rootTask, rank % numNodes, 1, &dep, 0);
        
        for (uint64_t rank = 0; rank < numNodes * numDummy; rank++)
            hiveEdtCreate(dummytask, rank % numNodes, 0, NULL, 0);
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (uint64_t) atoi(argv[1]);
    exitGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId)
    {
        if(!workerId)
        {
            PRINTF("Starting\n");
            hiveEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, 1);
            hiveGuid_t epochGuid = hiveInitializeAndStartEpoch(exitGuid, 0);
            hiveGuid_t startGuid = hiveEdtCreate(rootTask, 0, 1, &numDummy, 0);
        }
    }
}

int main(int argc, char** argv) 
{
    hiveRT(argc, argv);
    return 0;
}
