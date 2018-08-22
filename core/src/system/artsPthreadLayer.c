#define _GNU_SOURCE

#include <pthread.h>
#include "arts.h"
#include "artsMalloc.h"
#include "artsRuntime.h"
#include "artsConfig.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "limits.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsThreads.h"
#include <unistd.h>

unsigned int artsGlobalRankId;
unsigned int artsGlobalRankCount;
unsigned int artsGlobalMasterRankId;
struct artsConfig * gConfig;

artsGuid_t mainEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);

struct artsThreadArgs
{
    unsigned int coreId;
    unsigned int threadId;
    unsigned int uniqueThreadId;
};

pthread_t * nodeThreadList;
//struct artsThreadArgs * args;
struct artsConfig * config;

void *
artsThreadLoop(void * data)
{
    struct threadMask * unit = (struct threadMask*) data;
    if(unit->pin) {
        artsAbstractMachineModelPinThread(unit->coreInfo);
    }
    artsRuntimePrivateInit(unit, gConfig);
    artsRuntimeLoop();
    artsRuntimePrivateCleanup();
    return NULL;
    //pthread_exit(NULL);
}


void
artsThreadMainJoin()
{
    artsRuntimeLoop();
    artsRuntimePrivateCleanup();
    int i;
    for (i = 1; i < artsNodeInfo.totalThreadCount; i++)
        pthread_join(nodeThreadList[i], NULL);
    artsRuntimeGlobalCleanup();
    //artsFree(args);
    artsFree(nodeThreadList);
}

void
artsThreadInit( struct artsConfig * config  )
{
    gConfig = config;
    struct threadMask * mask = getThreadMask(config);
    nodeThreadList = artsMalloc(sizeof (pthread_t) * artsNodeInfo.totalThreadCount);
    unsigned int i = 0, threadCount=artsNodeInfo.totalThreadCount;
    if(config->stackSize)
    {
        void * stack;
        pthread_attr_t attr;
        long pageSize = sysconf(_SC_PAGESIZE);
        size_t size = ((config->stackSize%pageSize > 0) + (config->stackSize/pageSize))*pageSize;
        for (i=1; i<threadCount; i++)
        {
            pthread_attr_init(&attr);
            pthread_attr_setstacksize(&attr,size);
            pthread_create(&nodeThreadList[i], &attr, &artsThreadLoop, &mask[i]);
        }
    }
    else
    {
        for (i=1; i<threadCount; i++)
            pthread_create(&nodeThreadList[i], NULL, &artsThreadLoop, &mask[i]);
    }
    if(mask->pin)
        artsAbstractMachineModelPinThread(mask->coreInfo);
    artsRuntimePrivateInit(&mask[0], config);
}

void
artsShutdown()
{
    if(artsGlobalRankCount>1)
        artsRemoteShutdown();

    if(artsGlobalRankCount==1)
        artsRuntimeStop();
}

void
artsAbort(uint8_t errorCode)
{
    artsRemoteShutdown();

    if(artsGlobalRankCount==1)
        artsRuntimeStop();

    PRINTF("Abort: %d\n", errorCode);
}

uint64_t
getArgc(void *dbPtr)
{
    int *val = dbPtr;
    return (uint64_t) * val;
}

char *
getArgv(void *dbPtr, uint64_t count)
{
    int *argcAddress = dbPtr;
    char ***argvAddress = (char ***) (argcAddress + 1);
    return (*argvAddress)[count];
}

void artsThreadSetOsThreadCount(unsigned int threads)
{
    pthread_setconcurrency(threads);
}
