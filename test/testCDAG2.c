#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
unsigned int numWrites = 0;
hiveGuid_t dbGuid;
hiveGuid_t * writeGuids;


hiveGuid_t writeTest(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
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
        hiveSignalEdtValue((hiveGuid_t) paramv[1], -1, 0);
    }
    else
    {
        for(unsigned int i=0; i<numWrites; i++)
        {
            PRINTF("i: %u %u\n", i, array[i]);
        }
        hiveShutdown();
    }
        
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = hiveReserveGuidRoute(HIVE_DB_READ, 0);
    
    numWrites = atoi(argv[1]);
    writeGuids = hiveMalloc(sizeof(hiveGuid_t)*numWrites);
    for(unsigned int i=0; i<numWrites; i++)
        writeGuids[i] = hiveReserveGuidRoute(HIVE_EDT, i % hiveGetTotalNodes());
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        if(!nodeId)
        {
            unsigned int * ptr = hiveDbCreateWithGuid(dbGuid, sizeof(unsigned int) * numWrites);
            for(unsigned int i=0; i<numWrites; i++)
                ptr[i] = 0;
        }
        
        u64 args[2];
        for(u64 i=0; i<numWrites; i++)
        {
            if(hiveIsGuidLocal(writeGuids[i]))
            {
                args[0] = i;
                
                if(i < numWrites-1)
                {
                    args[1] = writeGuids[i+1];
                    hiveEdtCreateWithGuid(writeTest, writeGuids[i], 2, args, 2);
                }
                else
                {
                    hiveEdtCreateWithGuid(writeTest, writeGuids[i], 1, args, 2);
                }
                hiveSignalEdt(writeGuids[i], 0, hiveGuidCast(dbGuid, HIVE_DB_WRITE));
            }
        }
        if(!nodeId)
            hiveSignalEdtValue(writeGuids[0], -1, 0);
        
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}

