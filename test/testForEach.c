#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "artsRT.h"

unsigned int elemsPerNode = 4;
artsArrayDb_t * array = NULL;

artsGuid_t shutdown(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    PRINTF("Depc: %u\n", depc);
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elemsPerNode; j++)
        {
            PRINTF("%u: %u\n", i*elemsPerNode+j, data[j]);
        }
    }
    artsShutdown();
}

artsGuid_t check(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    artsGatherArrayDb(array, shutdown, 0, 0, NULL, 0);
}

artsGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    artsGuid_t checkGuid = paramv[1];
    unsigned int * value = depv[0].ptr;
    *value = index;
    PRINTF("%u:  %u %p\n", index, *value, value);
    artsSignalEdtValue(checkGuid, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        artsGuid_t checkGuid = artsEdtCreate(check, 0, 0, NULL, elemsPerNode * artsGetTotalNodes());
        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elemsPerNode * artsGetTotalNodes());
        artsForEachInArrayDbAtData(array, 1, edtFunc, 1, &checkGuid);
//        artsForEachInArrayDb(array, edtFunc, 1, &checkGuid);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
