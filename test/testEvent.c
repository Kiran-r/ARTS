#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbGuid = NULL_GUID;

hiveGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("HELLO\n");
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = hiveReserveGuidRoute(HIVE_DB_READ, hiveGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(hiveGetTotalNodes() - 1 == nodeId)
    {
        hiveDbCreateWithGuid(dbGuid, sizeof(unsigned int));
    }
    
    if(nodeId != hiveGetTotalNodes() - 1 && !workerId)
    {
        hiveActiveMessageWithDbAt(edtFunc, 0, NULL, 0, dbGuid, hiveGetTotalNodes() - 1);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}

