#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
unsigned int numWrites = 0;
artsGuid_t dbGuid;
artsGuid_t * writeGuids;


void writeTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    unsigned int * array = depv[0].ptr;
//    if(array)
//    {
        for(unsigned int i=index; i<numWrites; i++)
            array[i] = index;
//    }
    if(paramc > 1)
    {
        PRINTF("-----------------SIGNALLING NEXT %u\n", index);
        artsSignalEdtValue((artsGuid_t) paramv[1], -1, 0);
    }
    else
    {
        for(unsigned int i=0; i<numWrites; i++)
        {
            PRINTF("i: %u %u\n", i, array[i]);
        }
        artsShutdown();
    } 
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    
    numWrites = atoi(argv[1]);
    writeGuids = artsMalloc(sizeof(artsGuid_t)*numWrites);
    for(unsigned int i=0; i<numWrites; i++)
        writeGuids[i] = artsReserveGuidRoute(ARTS_EDT, i % artsGetTotalNodes());
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        if(!nodeId)
        {
            unsigned int * ptr = artsDbCreateWithGuid(dbGuid, sizeof(unsigned int) * numWrites);
            for(unsigned int i=0; i<numWrites; i++)
                ptr[i] = 0;
        }
        
        uint64_t args[2];
        for(uint64_t i=0; i<numWrites; i++)
        {
            if(artsIsGuidLocal(writeGuids[i]))
            {
                args[0] = i;
                
                if(i < numWrites-1)
                {
                    args[1] = writeGuids[i+1];
                    artsEdtCreateWithGuid(writeTest, writeGuids[i], 2, args, 2);
                }
                else
                {
                    artsEdtCreateWithGuid(writeTest, writeGuids[i], 1, args, 2);
                }
                artsSignalEdt(writeGuids[i], 0, artsGuidCast(dbGuid, ARTS_DB_WRITE));
            }
        }
        if(!nodeId)
            artsSignalEdtValue(writeGuids[0], -1, 0);
        
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

