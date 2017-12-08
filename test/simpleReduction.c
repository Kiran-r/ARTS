#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

unsigned int numDbs = 0;
hiveGuid_t reductionGuid = NULL_GUID;

hiveGuid_t reduction(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    u64 total = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        int * dbPtr = depv[i].ptr;
        total+=dbPtr[0];
    }
    hiveSignalEdt(paramv[0], total, 0, DB_MODE_SINGLE_VALUE);
}

hiveGuid_t shutDown(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("Result %lu\n", depv[0].guid);
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numDbs = atoi(argv[1]);  
    reductionGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<numDbs; i++)
        {
            if(i % hiveGetTotalNodes() == nodeId)
            {
                int * ptr;
                hiveGuid_t dbGuid = hiveDbCreate((void**)&ptr, sizeof(unsigned int));
                ptr[0] = i;
                hiveSignalEdt(reductionGuid, dbGuid, i, DB_MODE_NON_COHERENT_READ);
            }
        }
        
        if(!nodeId)
        {
            hiveGuid_t guid = hiveEdtCreate(shutDown, 0, 0, NULL, 1, NULL);
            hiveEdtCreateWithGuid(reduction, reductionGuid, 1, (u64*)&guid, numDbs, NULL);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}