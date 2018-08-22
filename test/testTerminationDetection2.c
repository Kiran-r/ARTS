#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"
#include "artsAtomics.h"

artsArrayDb_t * array = NULL;
artsGuid_t arrayGuid = NULL_GUID;
unsigned int elements = 32;

artsGuid_t check(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        PRINTF("%u: %u\n", i, *data);
    }
    
    artsShutdown();
}

artsGuid_t gatherTask(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    PRINTF("Gather task\n");
    artsGuid_t edtGuid = artsEdtCreate(check, 0, 0, NULL, elements);
    for(unsigned int i=0; i<elements; i++)
        artsGetFromArrayDb(edtGuid, i, array, i);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    elements = (unsigned int) atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId) {
        artsGuid_t gatherGuid = artsEdtCreate(gatherTask, 0, 0, 0, 1);
        artsInitializeAndStartEpoch(gatherGuid, 0);
        arrayGuid = artsNewArrayDb(&array, sizeof(unsigned int), elements);
        for(unsigned int i=0; i<elements; i++)
        {
            artsPutInArrayDb(&i, NULL_GUID, 0, array, i);
        }
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
