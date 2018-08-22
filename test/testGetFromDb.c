#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

artsGuid_t dbGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
artsGuid_t edtGuidFixed = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;
unsigned int stride = 0;

void getter(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int sum = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<stride; j++)
        {
            sum+=data[j];
        }
    }
    artsSignalEdtValue(shutdownGuid, artsGetCurrentNode(), sum);
}

void creater(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * data = artsMalloc(sizeof(unsigned int)*numElements);
    for(unsigned int i=0; i<numElements; i++)
    {
        data[i] = i;
    }
    artsDbCreateWithGuidAndData(dbGuid, data, sizeof(unsigned int) * numElements);
    artsEdtCreateWithGuid(getter, edtGuidFixed, 0, NULL, blockSize/stride);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int sum = 0;
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        sum += (unsigned int)depv[i].guid;
    
    unsigned int compare = 0;
    for(unsigned int i=0; i<numElements; i++)
        compare += i;
    
    if(sum == compare)
        PRINTF("CHECK SUM: %u vs %u\n", sum, compare);
    else
        PRINTF("FAIL SUM: %u vs %u\n", sum, compare);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
    edtGuidFixed = artsReserveGuidRoute(ARTS_EDT, 0);
    numElements = atoi(argv[1]);
    blockSize = numElements / artsGetTotalNodes();
    stride = atoi(argv[2]);
    if(!nodeId)
        PRINTF("numElements: %u blockSize: %u stride: %u\n", numElements, blockSize, stride);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(blockSize % stride)
    {
        if(!nodeId && !workerId)
        {
            artsShutdown();
        }
        return;
    }
    
    if(!workerId)
    {
        if(!nodeId)
        {
            artsEdtCreate(creater, 0, 0, NULL, 0);
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes());
        }
        
        unsigned int deps = blockSize/stride;
        if(!nodeId)
        {
            for(unsigned int j=0; j<deps; j++)
            {
                artsGetFromDb(edtGuidFixed, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
        else
        {
            artsGuid_t edtGuid = artsEdtCreate(getter, nodeId, 0, NULL, deps);
            for(unsigned int j=0; j<deps; j++)
            {
                artsGetFromDb(edtGuid, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}