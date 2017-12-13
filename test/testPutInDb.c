#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbGuid = NULL_GUID;
hiveGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;
unsigned int stride = 0;

hiveGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    bool pass = true;
    if(hiveIsGuidLocal(depv[0].guid))
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
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    shutdownGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    numElements = atoi(argv[1]);
    blockSize = numElements / hiveGetTotalNodes();
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
            hiveShutdown();
        }
        return;
    }
    
    if(!workerId)
    {   
        unsigned int deps = blockSize/stride;
        for(unsigned int j=0; j<deps; j++)
        {
            unsigned int * data = hiveMalloc(sizeof(unsigned int) * stride);
            for(unsigned int i=0; i<stride; i++)
                data[i] = nodeId*blockSize + j*stride + i;
//            PRINTF("PUT: index: %u slot: %u\n", nodeId*blockSize + j*stride, nodeId*deps + j);
            hivePutInDb(data, shutdownGuid, dbGuid, nodeId*deps + j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            hiveFree(data);
        }
        
        if(!nodeId)
        {
            hiveDbCreateWithGuid(dbGuid, sizeof(unsigned int) * numElements, true);
            hiveEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, numElements/stride);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}