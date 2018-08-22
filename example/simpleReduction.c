#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

unsigned int numDbs = 0;
artsGuid_t reductionGuid = NULL_GUID;

artsGuid_t reduction(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    u64 total = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        int * dbPtr = depv[i].ptr;
        total+=dbPtr[0];
    }
    artsSignalEdtValue(paramv[0], 0, total);
}

artsGuid_t shutDown(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    PRINTF("Result %lu\n", depv[0].guid);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numDbs = artsGetTotalNodes();  
    reductionGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        int * ptr;
        artsGuid_t dbGuid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_READ);
        *ptr = nodeId;
        
        artsSignalEdt(reductionGuid, nodeId, dbGuid);
        
        if(!nodeId)
        {
            artsGuid_t guid = artsEdtCreate(shutDown, 0, 0, NULL, 1);
            artsEdtCreateWithGuid(reduction, reductionGuid, 1, (u64*)&guid, numDbs);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}