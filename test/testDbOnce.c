#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

artsGuid_t dbGuid = NULL_GUID;
artsGuid_t aGuid = NULL_GUID;
artsGuid_t bGuid = NULL_GUID;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * ptr;
    artsGuid_t guid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_ONCE);
    *ptr = 2;
    
    PRINTF("Check: %lu %u newGuid: %lu\n", depv[0].guid, *((unsigned int*)depv[0].ptr), guid);
    artsSignalEdt(bGuid, 0, guid);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("ShutdownEdt: %lu %u\n", depv[0].guid, *((unsigned int*)depv[0].ptr));
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_ONCE, 0);
    aGuid = artsReserveGuidRoute(ARTS_EDT, 1);
    bGuid = artsReserveGuidRoute(ARTS_EDT, 2);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        if(nodeId == 0)
        {
            unsigned int * aPtr = artsDbCreateWithGuid(dbGuid, sizeof(unsigned int));
            *aPtr = 1;
        }
        
        if(nodeId == 1)
        {
            artsEdtCreateWithGuid(check, aGuid, 0, NULL, 1);
            artsSignalEdt(aGuid, 0, dbGuid);
        }
        
        if(nodeId == 2)
        {
            artsEdtCreateWithGuid(shutDownEdt, bGuid, 0, NULL, 1);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}