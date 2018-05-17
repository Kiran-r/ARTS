#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

uint64_t numDummy = 0;

hiveGuid_t dummytask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{
    uint64_t index = paramv[0];
    uint64_t dep = paramv[1];
    PRINTF("Dep: %lu ID: %lu Current Node: %u Current Worker: %u\n", dep, index, hiveGetCurrentNode(), hiveGetCurrentWorker());
}

hiveGuid_t rootTask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) 
{    
    uint64_t dep = paramv[0];
    PRINTF("Root: %lu\n", dep);
    if(dep)
    {
        hiveGuid_t poolGuid = hiveInitializeAndStartEpoch(NULL_GUID, 0);
        
        dep--;
        unsigned int numNodes = hiveGetTotalNodes();
//        hiveEdtCreateShad(rootTask, (hiveGetCurrentNode()+1)%numNodes, 1, &dep);
        hiveEdtCreateDep(rootTask, (hiveGetCurrentNode()+1)%numNodes, 1, &dep, 0, false);
        
//        uint64_t args[2];
//        args[0] = dep;
//        
//        for(uint64_t i=0; i<numDummy; i++)
//        {
//            args[1] = i;
//            hiveEdtCreateDep(dummytask, i%numNodes, 2, args, 0, false);
//        }
        PRINTF("Waiting on %lu\n", poolGuid);
        if(hiveWaitOnHandle(poolGuid))
            PRINTF("Done waiting on %lu dep: %lu\n", poolGuid, dep);
    }
    
    if(dep+1 == numDummy)
        hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (uint64_t) atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId)
    {
        PRINTF("Starting\n");
        uint64_t arg = numDummy;
        hiveEdtCreateShad(rootTask, 0, 1, &arg);
    }
}

int main(int argc, char** argv) 
{
    hiveRT(argc, argv);
    return 0;
}
