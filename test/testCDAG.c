#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
unsigned int numReads = 0;
unsigned int numWrites = 0;
unsigned int numDynamicReads = 0;
unsigned int numDynamicWrites = 0;
hiveGuid_t shutdownGuid;
hiveGuid_t dbGuid;
hiveGuid_t * readGuids;
hiveGuid_t * writeGuids;

hiveGuid_t shutdownEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int * array = depv[0].ptr;
    for(unsigned int i=0; i<numWrites; i++)
    {
        PRINTF("i: %u %u\n", i, array[i]);
    }
    hiveShutdown();
    // Need to fix the return value
    return NULL_GUID;
}

hiveGuid_t readTest(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
//    PRINTF("READ\n");
    unsigned int * array = depv[0].ptr;
    for(unsigned int i=0; i<numWrites; i++)
    {
        if(array[i] != 0 && array[i] != i)
        {
            PRINTF("BAD VALUE i: %u %u\n", i, array[i]);
        }
    }
    hiveEdtEmptySignal(shutdownGuid);
    // Need to fix the return value
    return NULL_GUID;
}

hiveGuid_t writeTest(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    unsigned int * array = depv[0].ptr;
//    PRINTF("WRITE %u\n", index);
    array[index] += index;

    for(unsigned int i=0; i<numDynamicReads; i++)
    {
        hiveGuid_t guid = hiveEdtCreate(readTest, hiveGetCurrentNode(), 0, NULL, 1);
        hiveSignalEdt(guid, dbGuid, 0, DB_MODE_NON_COHERENT_READ);
    }

    for(unsigned int i=0; i<numDynamicWrites; i++)
    {
        paramv[0] = (paramv[0]+1) % numWrites;
        hiveGuid_t guid = hiveEdtCreate(readTest, hiveGetCurrentNode(), 0, NULL, 1);
        hiveSignalEdt(guid, dbGuid, 0, DB_MODE_NON_COHERENT_READ);
    }

    if(!index)
        hiveSignalEdt(shutdownGuid, depv[0].guid, 0, DB_MODE_NON_COHERENT_READ);
    else
        hiveEdtEmptySignal(shutdownGuid);
    // Need to fix the return value
    return NULL_GUID;
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numReads = atoi(argv[1]);
    numWrites = atoi(argv[2]);
    numDynamicReads = atoi(argv[3]);
    numDynamicWrites = atoi(argv[4]);
    if(!nodeId)
        PRINTF("Reads: %u Writes: %u Dynamic Reads: %u Dynamic Writes: %u Final Deps: %u\n", numReads, numWrites, numDynamicReads, numDynamicWrites, numDynamicReads*numWrites+numDynamicWrites*numWrites+numReads+numWrites);

    readGuids = hiveMalloc(sizeof(hiveGuid_t)*numReads);
    writeGuids = hiveMalloc(sizeof(hiveGuid_t)*numWrites);

    dbGuid = hiveReserveGuidRoute(HIVE_DB, 0);

    for(unsigned int i=0; i<numReads; i++)
        readGuids[i] = hiveReserveGuidRoute(HIVE_EDT, i % hiveGetTotalNodes());
    for(unsigned int i=0; i<numWrites; i++)
        writeGuids[i] = hiveReserveGuidRoute(HIVE_EDT, i % hiveGetTotalNodes());

    shutdownGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
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

            hiveEdtCreateWithGuid(shutdownEdt, shutdownGuid, 0, NULL, numDynamicReads*numWrites+numDynamicWrites*numWrites+numReads+numWrites);
        }

        for(u64 i=0; i<numReads; i++)
        {
            if(hiveIsGuidLocal(readGuids[i]))
            {
                hiveEdtCreateWithGuid(readTest, readGuids[i], 0, NULL, 1);
                hiveSignalEdt(readGuids[i], dbGuid, 0, DB_MODE_NON_COHERENT_READ);
            }
        }

        for(u64 i=0; i<numWrites; i++)
        {
            if(hiveIsGuidLocal(writeGuids[i]))
            {
                hiveEdtCreateWithGuid(writeTest, writeGuids[i], 1, &i, 1);
                hiveSignalEdt(writeGuids[i], dbGuid, 0, DB_MODE_CDAG_WRITE);
            }
        }


    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
