#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"
artsGuid_t shutdownGuid;
artsGuid_t * guids;

void shutdownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsShutdown();
}

void acquireTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * num = depv[i].ptr;
        printf("%u %u i: %u %u\n", artsGetCurrentNode(), artsGetCurrentWorker(), i, *num);
    }
    artsSignalEdtValue(shutdownGuid, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guids = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        guids[i] = artsReserveGuidRoute(ARTS_DB_READ, i);
        if(!nodeId)
            PRINTF("i: %u guid: %ld\n", i, guids[i]);
    }
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            if(artsIsGuidLocal(guids[i]))
            {
                unsigned int * ptr = artsDbCreateWithGuid(guids[i], sizeof(unsigned int));
                *ptr = i;
                PRINTF("Created i: %u guid: %ld\n", i, guids[i]);
            }
        }
        
        if(!nodeId)
        {
            artsEdtCreateWithGuid(shutdownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes()*artsGetTotalWorkers());     
        }
    }
    artsGuid_t edtGuid = artsEdtCreate(acquireTest, nodeId, 0, NULL, artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        artsSignalEdt(edtGuid, i, guids[i]);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

