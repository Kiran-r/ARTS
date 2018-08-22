#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

artsGuid_t dbGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;
unsigned int stride = 0;

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    bool pass = true;
    if(artsIsGuidLocal(depv[0].guid))
    {
        unsigned int * data = depv[0].ptr;
        for(unsigned int i=0; i<numElements; i++)
        {
            if(data[i]!=i)
            {
                PRINTF("FAIL %u vs %u\n", i, data[i]);
                pass = false;
            }
        }
    }
    if(pass)
        PRINTF("CHECK\n");
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
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
        unsigned int deps = blockSize/stride;
        for(unsigned int j=0; j<deps; j++)
        {
            unsigned int * data = artsMalloc(sizeof(unsigned int) * stride);
            for(unsigned int i=0; i<stride; i++)
                data[i] = nodeId*blockSize + j*stride + i;
//            PRINTF("PUT: index: %u slot: %u\n", nodeId*blockSize + j*stride, nodeId*deps + j);
            artsPutInDb(data, shutdownGuid, dbGuid, nodeId*deps + j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            artsFree(data);
        }
        
        if(!nodeId)
        {
            artsDbCreateWithGuid(dbGuid, sizeof(unsigned int) * numElements);
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, numElements/stride);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}