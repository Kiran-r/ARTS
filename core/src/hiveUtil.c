#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveRemote.h"
#include "hiveRemoteFunctions.h"
#include "hiveGlobals.h"
#include "hiveAtomics.h"
#include "hiveCounter.h"
#include "hiveRuntime.h"
#include "hiveEdtFunctions.h"
#include "hiveOutOfOrder.h"
#include "hiveRouteTable.h"
#include "hiveDebug.h"
#include "hiveEdtFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveUtil.h"
#include "hiveIntrospection.h"
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define DPRINTF( ... )

#ifndef NOPARALLEL
extern u64 globalGuidOn;
#endif

extern __thread struct hiveEdt * currentEdt;

hiveGuid_t hiveGetCurrentGuid()
{
    if(currentEdt)
    {
        return  currentEdt->currentEdt;
    }
    return NULL_GUID;
}

unsigned int hiveGetCurrentNode()
{
    return hiveGlobalRankId;
}

unsigned int hiveGetTotalNodes()
{
   return hiveGlobalRankCount;
}

unsigned int hiveGetTotalWorkers()
{
    return hiveNodeInfo.workerThreadCount;
}

unsigned int hiveGetCurrentWorker()
{
    return hiveThreadInfo.groupId;
}

void hiveStopLocalWorker()
{
    hiveThreadInfo.alive = false;
}

void hiveStopLocalNode()
{
    hiveRuntimeStop();
}

u64 threadSafeRandom()
{
    long int temp = 0;
    mrand48_r (&hiveThreadInfo.drand_buf, &temp);
    return (u64) temp;
}

//char * hiveParallelCreateMMAP( char * pathToFile)
//{
//    int fd, offset;
//    char *data;
//    struct stat sbuf;
//    if ((fd = open(pathToFile, O_RDONLY)) == -1)
//    {
//        perror("open");
//        exit(1);
//    }
//    if (stat(pathToFile, &sbuf) == -1)
//    {
//        perror("stat");
//        exit(1);
//    }
//
//    return mmap((caddr_t)0, sbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
//}
//
//void hiveParallelStartRead(hiveGuid_t * arrayOfGuids, unsigned int numberOfGuids, 
//             hiveRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(hiveIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%hiveNodeInfo.workerThreadCount == hiveThreadInfo.groupId)
//            {
//                if(file)
//                    fseek(file, 0, SEEK_SET);
//                else
//                    file = fopen(pathToFile, "r");
//                
//                if(file)
//                {
//                    void * dbTempPtr = NULL;
//                    unsigned int dbSize = 0;
//                    reader(file, i, &dbSize, &dbTempPtr);
//                    if(dbTempPtr && dbSize)
//                    {
//                        HIVESETMEMSHOTTYPE(hiveDbMemorySize);
//                        struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + dbSize);
//                        OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                        memcpy(ptr+1, dbTempPtr, dbSize);
//                        ocrFree(dbTempPtr);
//                        ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb)+dbSize, false);
//                        ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                        ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                    }
//                }
//                else
//                {
//                    PRINTF("Unable to open file %s\n", pathToFile);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//    if(file)
//        fclose(file);
//}
//
//void ocrParallelStartReadFixedSizeMMAP(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids, 
//                              unsigned int size, ocrRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                {
//                    OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                    struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + size);
//                    OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                    void * temp = ptr + 1;
//                    unsigned int dbSize = size;
//                    reader(pathToFile, i, &dbSize, &temp);
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//}
//
//
//void ocrParallelStartReadFixedSizeLine(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids, unsigned int size, ocrRecordReaderLine_t reader, char * pathToFile, ocrMapper_t mapper, unsigned int mapperLength)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    unsigned int MAX_BUFFER_LENGTH = 128;
//    file = fopen(pathToFile, "r");
//    char buffer[MAX_BUFFER_LENGTH];
//    unsigned int localDbArrayLength=0;
//    
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//                localDbArrayLength++;
//            blocksPerNode++;
//        }
//    }
//    struct ocrDb ** localDbArray = ocrMalloc(sizeof(struct ocrDb*) * localDbArrayLength);
//    unsigned int * localDbArrayMap = ocrMalloc(sizeof(unsigned int) * localDbArrayLength*mapperLength);
//    unsigned int localOffset=0;
//    blocksPerNode=0;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                localDbArray[localOffset] = ocrCalloc(sizeof(struct ocrDb) + size);
//                for(int j=0; j< mapperLength; j++)
//                    localDbArrayMap[localOffset*mapperLength+j] = mapper(i, j);
//                OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                localOffset++;
//            }
//            blocksPerNode++;
//        }
//    }
//    bool firstRead=0;
//    if(file)
//    {
//        while (fgets(buffer, MAX_BUFFER_LENGTH, file))
//        {
//            localOffset=0;
//            blocksPerNode=0;
//            firstRead=true;
//            for(unsigned int i=0; i<localDbArrayLength; i++)
//            {
//                struct ocrDb * ptr = localDbArray[i];
//                void * temp = ptr + 1;
//                unsigned int dbSize = size;
//                if(firstRead)
//                    reader(buffer, i, &dbSize, &temp, localDbArrayMap+i*mapperLength);
//                else
//                    reader(NULL, i, &dbSize, &temp, localDbArrayMap+i*mapperLength);
//                 
//                firstRead = false;
//            }
//        }
//        fclose(file);
//        localOffset=0;
//        blocksPerNode=0;
//        for(unsigned int i=0; i<numberOfGuids; i++)
//        {
//            if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//            {
//                if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//                {
//                    struct ocrDb * ptr = localDbArray[localOffset];
//                    unsigned int dbSize = size;
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                    localOffset++;
//                }
//                blocksPerNode++;
//            }
//        }
//    }
//    else
//    {
//        PRINTF("Unable to open file %s\n", pathToFile);
//    }
//    ocrFree(localDbArray);
//    ocrFree(localDbArrayMap);
//}
//
//void ocrParallelStartReadFixedSize(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids, 
//                              unsigned int size, ocrRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                if(file)
//                    fseek(file, 0, SEEK_SET);
//                else
//                    file = fopen(pathToFile, "r");
//                if(file)
//                {
//                    OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                    struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + size);
//                    OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                    void * temp = ptr + 1;
//                    unsigned int dbSize = size;
//                    reader(file, i, &dbSize, &temp);
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                }
//                else
//                {
//                    PRINTF("Unable to open file %s\n", pathToFile);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//    if(file)
//        fclose(file);
//}
