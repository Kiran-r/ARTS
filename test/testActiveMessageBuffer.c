#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbDestGuid = NULL_GUID;
hiveGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

hiveGuid_t setter(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    
    unsigned int id = paramv[0];
    unsigned int * buffer = depv[0].ptr;
    unsigned int * dest = depv[1].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        dest[id*blockSize + i] = buffer[i];
    }
    PRINTF("Setter: %u\n", id);
    hiveSignalEdt(shutdownGuid, id, dbDestGuid);
}

hiveGuid_t getter(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int id = paramv[0];
    unsigned int * source = depv[0].ptr;
    unsigned int * buffer = &source[id*blockSize];
    PRINTF("Getter: %u\n", id);
    //This one actually sends to a remote node... yea for testing!
    hiveGuid_t am = hiveActiveMessageWithBuffer(setter, hiveGetTotalNodes() - 1, paramc, paramv, 1, buffer, sizeof(unsigned int)*blockSize);
    hiveSignalEdt(am, 1, dbDestGuid);
}

hiveGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    bool pass = true;
    unsigned int * data = depv[0].ptr;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(data[i]!=i)
        {
            PRINTF("I: %u vs %u\n", i, data[i]);
            pass = false;
        }
    }
    
    if(pass)
        PRINTF("CHECK\n");
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    blockSize = atoi(argv[1]);
    numElements = blockSize * hiveGetTotalNodes();
    dbDestGuid = hiveReserveGuidRoute(HIVE_DB_PIN, hiveGetTotalNodes() - 1);
    shutdownGuid = hiveReserveGuidRoute(HIVE_EDT, hiveGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {   
        u64 id = nodeId;
        unsigned int * data = hiveMalloc(sizeof(unsigned int)*numElements);
        for(unsigned int i=0; i<numElements; i++)
        {
            data[i] = i;
        }
        //This is kinda dumb since it is sending to itself, but hey lets check it...
        hiveActiveMessageWithBuffer(getter, nodeId, 1, &id, 0, data, sizeof(unsigned int)*numElements);
        
        if(!nodeId)
            hiveEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, hiveGetTotalNodes());
        
        if(nodeId == hiveGetTotalNodes() - 1)
            hiveDbCreateWithGuid(dbDestGuid, sizeof(unsigned int) * numElements);
    }
}


int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}