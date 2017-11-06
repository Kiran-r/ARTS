#define _GNU_SOURCE

#include <pthread.h>
#include "hive.h"
#include "hiveMalloc.h"
#include "hiveRuntime.h"
#include "hiveConfig.h"
#include "hiveRemote.h"
#include "hiveGuid.h"
#include "limits.h"
#include "hiveGlobals.h"
#include "hiveCounter.h"
#include "hiveThreads.h"
#include <unistd.h>

unsigned int hiveGlobalRankId;
unsigned int hiveGlobalRankCount;
unsigned int hiveGlobalMasterRankId;
struct hiveConfig * gConfig;

hiveGuid_t mainEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

struct hiveThreadArgs
{
    unsigned int coreId;
    unsigned int threadId;
    unsigned int uniqueThreadId;
};

pthread_t * nodeThreadList;
//struct hiveThreadArgs * args;
struct hiveConfig * config;

void
hiveThreadPin(void * data)
{
    unsigned int * id = data;  
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(*id, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}


void *
hiveThreadLoop(void * data)
{
    u64 actualStackSize;
    pthread_attr_t tattr;
    pthread_getattr_np(pthread_self(), &tattr);
    pthread_attr_getstacksize(&tattr, &actualStackSize);
    struct threadMask * unit = (struct threadMask*) data;
    if(unit->pin)
        hiveAbstractMachineModelPinThread(unit->coreInfo);
        //hiveThreadPin(unit->unitId);
    hiveRuntimePrivateInit(unit, gConfig);
    hiveRuntimeLoop();
    hiveRuntimePrivateCleanup();
    return NULL;
    //pthread_exit(NULL);
}


void
hiveThreadMainJoin()
{
    hiveRuntimeLoop();
    hiveRuntimePrivateCleanup();
    int i;
    for (i = 1; i < hiveNodeInfo.totalThreadCount; i++)
        pthread_join(nodeThreadList[i], NULL);
    hiveRuntimeGlobalCleanup();
    //hiveFree(args);
    hiveFree(nodeThreadList);
}

void
hiveThreadInit( struct hiveConfig * config  )
{
    gConfig = config;
    struct threadMask * mask = getThreadMask(config);    
    nodeThreadList = hiveMalloc(sizeof (pthread_t) * hiveNodeInfo.totalThreadCount);
    unsigned int i, threadCount=hiveNodeInfo.totalThreadCount;
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
            pthread_create(&nodeThreadList[i], &attr, &hiveThreadLoop, &mask[i]);
        }
    }
    else
    {
        for (i=1; i<threadCount; i++)
            pthread_create(&nodeThreadList[i], NULL, &hiveThreadLoop, &mask[i]);
    }
    if(mask->pin)
        hiveAbstractMachineModelPinThread(mask->coreInfo);
    hiveRuntimePrivateInit(&mask[0], config);
}

void
hiveShutdown()
{
    if(hiveGlobalRankCount>1)
        hiveRemoteShutdown();
    
    if(hiveGlobalRankCount==1)
        hiveRuntimeStop();
}

void
hiveAbort(u8 errorCode)
{
    hiveRemoteShutdown();
    
    if(hiveGlobalRankCount==1)
        hiveRuntimeStop();

    PRINTF("Abort: %d\n", errorCode);
}

u64
getArgc(void *dbPtr)
{
    int *val = dbPtr;
    return (u64) * val;
}

char *
getArgv(void *dbPtr, u64 count)
{
    int *argcAddress = dbPtr;
    char ***argvAddress = (char ***) (argcAddress + 1);
    return (*argvAddress)[count];
}

void hiveThreadSetOsThreadCount(unsigned int threads)
{
    pthread_setconcurrency(threads);
}
