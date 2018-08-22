#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

artsGuid_t dbGuid = NULL_GUID;

artsGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    PRINTF("HELLO\n");
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_READ, artsGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(artsGetTotalNodes() - 1 == nodeId)
    {
        artsDbCreateWithGuid(dbGuid, sizeof(unsigned int));
    }
    
    if(nodeId != artsGetTotalNodes() - 1 && !workerId)
    {
        artsActiveMessageWithDbAt(edtFunc, 0, NULL, 0, dbGuid, artsGetTotalNodes() - 1);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

