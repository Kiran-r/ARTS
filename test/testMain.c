#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

void test(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Running edt %lu on %u %u\n", artsGetCurrentGuid(), artsGetCurrentNode(), artsGetCurrentWorker());
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    PRINTF("Node %u argc %u\n", nodeId, argc);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId && !workerId)
    {
        uint64_t args[3];
        artsGuid_t guid = artsEdtCreate(test, 0, 3, args, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}