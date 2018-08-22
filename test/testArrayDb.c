#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

unsigned int elements = 32;
artsArrayDb_t * array = NULL;

artsGuid_t check(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        PRINTF("%u: %u\n", i, *data);
    }
    
    artsShutdown();
}

artsGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    artsGuid_t edtGuid = artsEdtCreate(check, 0, 0, NULL, elements);
    for(unsigned int i=0; i<depc; i++)
        artsGetFromArrayDb(edtGuid, i, array, i);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    if(argc > 1)
        elements = atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!nodeId && !workerId)
    {
        artsGuid_t edtGuid = artsEdtCreate(edtFunc, 0, 0, NULL, elements);
        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elements);
        for(unsigned int i=0; i<elements; i++)
        {
            artsPutInArrayDb(&i, edtGuid, i, array, i);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
