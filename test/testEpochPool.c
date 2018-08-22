#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"
#include "shadAdapter.h"

uint64_t numDummy = 0;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    uint64_t index = paramv[0];
    uint64_t dep = paramv[1];
    PRINTF("Dep: %lu ID: %lu Current Node: %u Current Worker: %u\n", dep, index, artsGetCurrentNode(), artsGetCurrentWorker());
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{    
    uint64_t dep = paramv[0];
    PRINTF("Root: %lu\n", dep);
    if(dep)
    {
        artsGuid_t poolGuid = artsInitializeAndStartEpoch(NULL_GUID, 0);
        
        dep--;
        unsigned int numNodes = artsGetTotalNodes();
//        artsEdtCreateShad(rootTask, (artsGetCurrentNode()+1)%numNodes, 1, &dep);
        artsEdtCreateDep(rootTask, (artsGetCurrentNode()+1)%numNodes, 1, &dep, 0, false);
        
//        uint64_t args[2];
//        args[0] = dep;
//        
//        for(uint64_t i=0; i<numDummy; i++)
//        {
//            args[1] = i;
//            artsEdtCreateDep(dummytask, i%numNodes, 2, args, 0, false);
//        }
        PRINTF("Waiting on %lu\n", poolGuid);
        if(artsWaitOnHandle(poolGuid))
            PRINTF("Done waiting on %lu dep: %lu\n", poolGuid, dep);
    }
    
    if(dep+1 == numDummy)
        artsShutdown();
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
        artsEdtCreateShad(rootTask, 0, 1, &arg);
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
