#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t test(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("Running edt %lu on %u %u\n", hiveGetCurrentGuid(), hiveGetCurrentNode(), hiveGetCurrentWorker());
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    PRINTF("Node %u argc %u\n", nodeId, argc);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId && !workerId)
    {
        u64 args[3];
        hiveGuid_t guid = hiveEdtCreate(test, 0, 3, args, 0);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}