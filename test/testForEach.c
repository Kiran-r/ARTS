#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveArrayDb_t * array = NULL;

hiveGuid_t shutdown(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("Depc: %u\n", depc);
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<4; j++)
        {
            PRINTF("%u: %u\n", i*4+j, data[j]);
        }
    }
    hiveShutdown();
}

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveGatherArrayDb(array, shutdown, 0, 0, NULL, 0);
}

hiveGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    hiveGuid_t checkGuid = paramv[1];
    unsigned int * value = depv[0].ptr;
    *value = index;
    PRINTF("%u: %u %p\n", index, value, value);
    hiveSignalEdt(checkGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        hiveGuid_t checkGuid = hiveEdtCreate(check, 0, 0, NULL, 32);
        hiveGuid_t guid = hiveNewArrayDb(&array, sizeof(unsigned int), 32);
        hiveForEachInArrayDbAtData(array, 1, edtFunc, 1, &checkGuid);
//        hiveForEachInArrayDb(array, edtFunc, 1, &checkGuid);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
