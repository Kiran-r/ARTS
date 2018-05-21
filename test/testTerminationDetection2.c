#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "hiveAtomics.h"

hiveArrayDb_t * array = NULL;
hiveGuid_t arrayGuid = NULL_GUID;
unsigned int elements = 32;

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        PRINTF("%u: %u\n", i, *data);
    }
    
    hiveShutdown();
}

hiveGuid_t gatherTask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("Gather task\n");
    hiveGuid_t edtGuid = hiveEdtCreate(check, 0, 0, NULL, elements);
    for(unsigned int i=0; i<elements; i++)
        hiveGetFromArrayDb(edtGuid, i, array, i);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    elements = (unsigned int) atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId) {
        hiveGuid_t gatherGuid = hiveEdtCreate(gatherTask, 0, 0, 0, 1);
        hiveInitializeAndStartEpoch(gatherGuid, 0);
        arrayGuid = hiveNewArrayDb(&array, sizeof(unsigned int), elements);
        for(unsigned int i=0; i<elements; i++)
        {
            hivePutInArrayDb(&i, NULL_GUID, 0, array, i);
        }
    }
}

int main(int argc, char** argv) 
{
    hiveRT(argc, argv);
    return 0;
}
