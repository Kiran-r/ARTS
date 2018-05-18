#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t guid[4];
hiveGuid_t shutdownGuid = NULL_GUID;

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        for(unsigned int j=0; j<4; j++)
        {
            if(guid[j] == depv[i].guid)
            {
                unsigned int * data = depv[i].ptr;
                PRINTF("j: %u %lu: %u from %u\n", j, depv[i].guid, *data, hiveGuidGetRank(depv[i].guid));
            }
        }
    }
    hiveSignalEdt(shutdownGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
}

hiveGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guid[0] = hiveReserveGuidRoute(HIVE_DB, 0);
    guid[1] = hiveReserveGuidRoute(HIVE_DB, 0);
    guid[2] = hiveReserveGuidRoute(HIVE_DB, 1);
    guid[3] = hiveReserveGuidRoute(HIVE_DB, 1);
    
    shutdownGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        if(nodeId == 0)
        {
            //Local to local
            unsigned int * aPtr = hiveDbCreateWithGuid(guid[0], sizeof(unsigned int), false);
            *aPtr = 1;
            hiveDbMove(guid[0], 0);
            
            //Local to remote
            unsigned int * aPtr2 = hiveDbCreateWithGuid(guid[1], sizeof(unsigned int), false);
            *aPtr2 = 2;
            hiveDbMove(guid[1], 1);
            
            //Remote to local
            hiveDbMove(guid[2], 0);
            
            //Remote to remote
            hiveDbMove(guid[3], 2);
        }
        
        if(nodeId == 1)
        {
            unsigned int * bPtr = hiveDbCreateWithGuid(guid[2], sizeof(unsigned int), false);
            *bPtr = 3;
            
            unsigned int * cPtr = hiveDbCreateWithGuid(guid[3], sizeof(unsigned int), false);
            *cPtr = 4;
        }
    }
    if(!workerId)
    {
        if(nodeId == 0)
        {
            hiveGuid_t edtGuid = hiveEdtCreate(check, nodeId, 0, NULL, 2);
            hiveSignalEdt(edtGuid, guid[0], 0, DB_MODE_ONCE_LOCAL);
            hiveSignalEdt(edtGuid, guid[2], 1, DB_MODE_ONCE_LOCAL);
            
            hiveEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, 3);
        }
        
        if(nodeId == 1)
        {
            hiveGuid_t edtGuid = hiveEdtCreate(check, nodeId, 0, NULL, 1);
            hiveSignalEdt(edtGuid, guid[1], 0, DB_MODE_ONCE_LOCAL);
        }
        
        if(nodeId == 2)
        {
            hiveGuid_t edtGuid = hiveEdtCreate(check, nodeId, 0, NULL, 1);
            hiveSignalEdt(edtGuid, guid[3], 0, DB_MODE_ONCE_LOCAL);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}