#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

artsGuid_t guid[4];
artsGuid_t shutdownGuid = NULL_GUID;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        for(unsigned int j=0; j<4; j++)
        {
            if(guid[j] == depv[i].guid)
            {
                unsigned int * data = depv[i].ptr;
                PRINTF("j: %u %lu: %u from %u\n", j, depv[i].guid, *data, artsGuidGetRank(depv[i].guid));
            }
        }
    }
    artsSignalEdtValue(shutdownGuid, -1, 0);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guid[0] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 0);
    guid[1] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 0);
    guid[2] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 1);
    guid[3] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 1);
    
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        if(nodeId == 0)
        {
            //Local to local
            unsigned int * aPtr = artsDbCreateWithGuid(guid[0], sizeof(unsigned int));
            *aPtr = 1;
            artsDbMove(guid[0], 0);
            
            //Local to remote
            unsigned int * aPtr2 = artsDbCreateWithGuid(guid[1], sizeof(unsigned int));
            *aPtr2 = 2;
            artsDbMove(guid[1], 1);
            
            //Remote to local
            artsDbMove(guid[2], 0);
            
            //Remote to remote
            artsDbMove(guid[3], 2);
        }
        
        if(nodeId == 1)
        {
            unsigned int * bPtr = artsDbCreateWithGuid(guid[2], sizeof(unsigned int));
            *bPtr = 3;
            
            unsigned int * cPtr = artsDbCreateWithGuid(guid[3], sizeof(unsigned int));
            *cPtr = 4;
        }
    }
    if(!workerId)
    {
        if(nodeId == 0)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 2);
            artsSignalEdt(edtGuid, 0, guid[0]);
            artsSignalEdt(edtGuid, 1, guid[2]);
            
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, 3);
        }
        
        if(nodeId == 1)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 1);
            artsSignalEdt(edtGuid, 0, guid[1]);
        }
        
        if(nodeId == 2)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 1);
            artsSignalEdt(edtGuid, 0, guid[3]);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}