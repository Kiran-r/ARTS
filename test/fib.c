#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t fibJoin(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int x = depv[0].guid;
    unsigned int y = depv[1].guid;
    hiveSignalEdt(paramv[0], x+y, paramv[1], DB_MODE_SINGLE_VALUE);
}

hiveGuid_t fibFork(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveGuid_t guid = paramv[0];
    unsigned int slot = paramv[1];
    unsigned int num = paramv[2];
    if(num < 2)
        hiveSignalEdt(guid, num, slot, DB_MODE_SINGLE_VALUE);
    else
    {
        hiveGuid_t joinGuid = hiveEdtCreate(fibJoin, 0, paramc-1, paramv, 2, NULL);
        
        u64 args[3] = {joinGuid, 0, num-1};
        hiveEdtCreate(fibFork, 0, 3, args, 0, NULL);
        
        args[1] = 1;
        args[2] = num-2;
        hiveEdtCreate(fibFork, 0, 3, args, 0, NULL);
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
        hiveGuid_t doneGuid = hiveEdtCreate(fibDone, 0, 1, (u64*)&num, 1, NULL);
        u64 args[3] = {doneGuid, 0, num};
        hiveGuid_t guid = hiveEdtCreate(fibFork, 0, 3, args, 0, NULL);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}