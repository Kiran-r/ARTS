#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
hiveGuid_t shutdownGuid;
hiveGuid_t * guids;

hiveGuid_t shutdownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    hiveShutdown();
}

hiveGuid_t acquireTest(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * num = depv[i].ptr;
        printf("%u %u i: %u %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker(), i, *num);
    }
    hiveSignalEdt(shutdownGuid, 0, 0, DB_MODE_SINGLE_VALUE);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guids = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
    {
        guids[i] = hiveReserveGuidRoute(HIVE_DB, i);
        if(!nodeId)
            PRINTF("i: %u guid: %ld\n", i, guids[i]);
    }
    shutdownGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        {
            if(hiveIsGuidLocal(guids[i]))
            {
                unsigned int * ptr = hiveDbCreateWithGuid(guids[i], sizeof(unsigned int));
                *ptr = i;
                PRINTF("Created i: %u guid: %ld\n", i, guids[i]);
            }
        }
        
        if(!nodeId)
        {
            hiveEdtCreateWithGuid(shutdownEdt, shutdownGuid, 0, NULL, hiveGetTotalNodes()*hiveGetTotalWorkers(), NULL);     
        }
    }
    hiveGuid_t edtGuid = hiveEdtCreate(acquireTest, nodeId, 0, NULL, hiveGetTotalNodes(), NULL);
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
    {
        hiveSignalEdt(edtGuid, guids[i], i, DB_MODE_NON_COHERENT_READ);
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}

