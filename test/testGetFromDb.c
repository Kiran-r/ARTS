#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbGuid = NULL_GUID;
hiveGuid_t shutdownGuid = NULL_GUID;
hiveGuid_t edtGuidFixed = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;
unsigned int stride = 0;

hiveGuid_t getter(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
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
    hiveSignalEdt(shutdownGuid, sum, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);
}

hiveGuid_t creater(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int * data = hiveMalloc(sizeof(unsigned int)*numElements);
    for(unsigned int i=0; i<numElements; i++)
    {
        data[i] = i;
    }
    hiveDbCreateWithGuidAndData(dbGuid, data, sizeof(unsigned int) * numElements, true);
    hiveEdtCreateWithGuid(getter, edtGuidFixed, 0, NULL, blockSize/stride);
}

hiveGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int sum = 0;
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        sum += (unsigned int)depv[i].guid;
    
    unsigned int compare = 0;
    for(unsigned int i=0; i<numElements; i++)
        compare += i;
    
    if(sum == compare)
        PRINTF("CHECK SUM: %u vs %u\n", sum, compare);
    else
        PRINTF("FAIL SUM: %u vs %u\n", sum, compare);
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    shutdownGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    edtGuidFixed = hiveReserveGuidRoute(HIVE_EDT, 0);
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
        if(!nodeId)
        {
            hiveEdtCreate(creater, 0, 0, NULL, 0);
            hiveEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, hiveGetTotalNodes());
        }
        
        unsigned int deps = blockSize/stride;
        if(!nodeId)
        {
            for(unsigned int j=0; j<deps; j++)
            {
                hiveGetFromDb(edtGuidFixed, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
        else
        {
            hiveGuid_t edtGuid = hiveEdtCreate(getter, nodeId, 0, NULL, deps);
            for(unsigned int j=0; j<deps; j++)
            {
                hiveGetFromDb(edtGuid, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}