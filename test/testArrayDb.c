#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveArrayDb_t * array = NULL;

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        PRINTF("%u: %u\n", i, *data);
    }
    
    hiveShutdown();
}

hiveGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveGuid_t edtGuid = hiveEdtCreate(check, 0, 0, NULL, 32);
    for(unsigned int i=0; i<depc; i++)
        hiveGetFromArrayDb(edtGuid, i, array, i);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!nodeId && !workerId)
    {
        hiveGuid_t edtGuid = hiveEdtCreate(edtFunc, 0, 0, NULL, 32);
        hiveGuid_t guid = hiveNewArrayDb(&array, sizeof(unsigned int), 4, 8);
        for(unsigned int i=0; i<32; i++)
        {
            unsigned int * temp = hiveMalloc(sizeof(unsigned int));
            *temp = i;
            hivePutInArrayDb(temp, edtGuid, i, array, i);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
