#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbGuid = NULL_GUID;
hiveGuid_t aGuid = NULL_GUID;
hiveGuid_t bGuid = NULL_GUID;

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int * ptr;
    hiveGuid_t guid = hiveDbCreate((void**)&ptr, sizeof(unsigned int), HIVE_DB_ONCE);
    *ptr = 2;
    
    PRINTF("Check: %lu %u newGuid: %lu\n", depv[0].guid, *((unsigned int*)depv[0].ptr), guid);
    hiveSignalEdt(bGuid, guid, 0, DB_MODE_ONCE);
}

hiveGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("ShutdownEdt: %lu %u\n", depv[0].guid, *((unsigned int*)depv[0].ptr));
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    aGuid = hiveReserveGuidRoute(HIVE_EDT, 1);
    bGuid = hiveReserveGuidRoute(HIVE_EDT, 2);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        if(nodeId == 0)
        {
            unsigned int * aPtr = hiveDbCreateWithGuid(dbGuid, sizeof(unsigned int));
            *aPtr = 1;
        }
        
        if(nodeId == 1)
        {
            hiveEdtCreateWithGuid(check, aGuid, 0, NULL, 1);
            hiveSignalEdt(aGuid, dbGuid, 0, DB_MODE_ONCE);
        }
        
        if(nodeId == 2)
        {
            hiveEdtCreateWithGuid(shutDownEdt, bGuid, 0, NULL, 1);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}