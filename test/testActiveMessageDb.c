#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

hiveGuid_t dbSourceGuid = NULL_GUID;
hiveGuid_t dbDestGuid = NULL_GUID;
hiveGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

hiveGuid_t setter(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    
    unsigned int id = paramv[0];
    unsigned int * dest = depv[0].ptr;
    unsigned int * buffer = depv[1].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        dest[id*blockSize + i] = buffer[i];
    }
    hiveSignalEdt(shutdownGuid, dbDestGuid, id, DB_MODE_PIN);
}

hiveGuid_t getter(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int * buffer;
    hiveGuid_t cpyDb = hiveDbCreate((void **) &buffer, sizeof(unsigned int)*blockSize, false);
    
    unsigned int id = paramv[0];
    unsigned int * source = depv[0].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        buffer[i] = source[id*blockSize + i];
    }
    hiveGuid_t am = hiveActiveMessageWithDb(setter, paramc, paramv, 1, dbDestGuid);
    hiveSignalEdt(am, cpyDb, 1, DB_MODE_NON_COHERENT_READ);
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
    dbSourceGuid = hiveReserveGuidRoute(HIVE_DB, 0);
    dbDestGuid = hiveReserveGuidRoute(HIVE_DB, hiveGetTotalNodes() - 1);
    shutdownGuid = hiveReserveGuidRoute(HIVE_DB, hiveGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {   
        u64 id = nodeId;
        hiveActiveMessageWithDb(getter, 1, &id, 0, dbSourceGuid);
        
        if(!nodeId)
        {
            unsigned int * data = hiveMalloc(sizeof(unsigned int)*numElements);
            for(unsigned int i=0; i<numElements; i++)
            {
                data[i] = i;
            }
            hiveDbCreateWithGuidAndData(dbSourceGuid, data, sizeof(unsigned int) * numElements, true);
            hiveEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, hiveGetTotalNodes());
        }
        
        if(nodeId == hiveGetTotalNodes() - 1)
            hiveDbCreateWithGuid(dbDestGuid, sizeof(unsigned int) * numElements, true);
    }
}


int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}