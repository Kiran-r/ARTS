#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

artsGuid_t dbSourceGuid = NULL_GUID;
artsGuid_t dbDestGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

artsGuid_t setter(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    
    unsigned int id = paramv[0];
    unsigned int * dest = depv[0].ptr;
    unsigned int * buffer = depv[1].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        dest[id*blockSize + i] = buffer[i];
    }
    artsSignalEdt(shutdownGuid, id, dbDestGuid);
}

artsGuid_t getter(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
{
    unsigned int * buffer;
    artsGuid_t cpyDb = artsDbCreate((void **) &buffer, sizeof(unsigned int)*blockSize, ARTS_DB_READ);
    
    unsigned int id = paramv[0];
    unsigned int * source = depv[0].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        buffer[i] = source[id*blockSize + i];
    }
    artsGuid_t am = artsActiveMessageWithDb(setter, paramc, paramv, 1, dbDestGuid);
    artsSignalEdt(am, 1, cpyDb);
}


artsGuid_t shutDownEdt(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[])
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
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    blockSize = atoi(argv[1]);
    numElements = blockSize * artsGetTotalNodes();
    dbSourceGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    dbDestGuid = artsReserveGuidRoute(ARTS_DB_READ, artsGetTotalNodes() - 1);
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, artsGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {   
        u64 id = nodeId;
        artsActiveMessageWithDb(getter, 1, &id, 0, dbSourceGuid);
        
        if(!nodeId)
        {
            unsigned int * data = artsMalloc(sizeof(unsigned int)*numElements);
            for(unsigned int i=0; i<numElements; i++)
            {
                data[i] = i;
            }
            artsDbCreateWithGuidAndData(dbSourceGuid, data, sizeof(unsigned int) * numElements);
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes());
        }
        
        if(nodeId == artsGetTotalNodes() - 1)
            artsDbCreateWithGuid(dbDestGuid, sizeof(unsigned int) * numElements);
    }
}


int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}