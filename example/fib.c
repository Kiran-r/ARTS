#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t fibJoin(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int x = depv[0].guid;
    unsigned int y = depv[1].guid;
    hiveSignalEdtValue(paramv[0], paramv[1], x+y);
}

hiveGuid_t fibFork(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int next = (hiveGetCurrentNode() + 1) % hiveGetTotalNodes();
//    PRINTF("NODE: %u WORKER: %u NEXT: %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker(), next);
    hiveGuid_t guid = paramv[0];
    unsigned int slot = paramv[1];
    unsigned int num = paramv[2];
    if(num < 2)
        hiveSignalEdtValue(guid, slot, num);
    else
    {
        hiveGuid_t joinGuid = hiveEdtCreate(fibJoin, 0, paramc-1, paramv, 2);
        
        u64 args[3] = {joinGuid, 0, num-1};
        hiveEdtCreate(fibFork, next, 3, args, 0);
        
        args[1] = 1;
        args[2] = num-2;
        hiveEdtCreate(fibFork, next, 3, args, 0);
    }
}

hiveGuid_t fibDone(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("Fib %u: %u %u\n", paramv[0], depv[0].guid, depc);
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        unsigned int num = atoi(argv[1]);
        hiveGuid_t doneGuid = hiveEdtCreate(fibDone, 0, 1, (u64*)&num, 1);
        u64 args[3] = {doneGuid, 0, num};
        hiveGuid_t guid = hiveEdtCreate(fibFork, 0, 3, args, 0);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}