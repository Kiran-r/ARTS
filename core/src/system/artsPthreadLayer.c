#define _GNU_SOURCE
#include <pthread.h>
#include "arts.h"
#include "artsRuntime.h"
#include "artsConfig.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "limits.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsThreads.h"
#include "artsTMT.h"
#include <unistd.h>

unsigned int artsGlobalRankId;
unsigned int artsGlobalRankCount;
unsigned int artsGlobalMasterRankId;
struct artsConfig * gConfig;
struct artsConfig * config;

pthread_t * nodeThreadList;

void * artsThreadLoop(void * data)
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


void artsThreadMainJoin()
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

void artsThreadInit( struct artsConfig * config  )
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

void artsShutdown()
{
    if(artsGlobalRankCount>1)
        artsRemoteShutdown();

    if(artsGlobalRankCount==1)
        artsRuntimeStop();
}

void artsThreadSetOsThreadCount(unsigned int threads)
{
    pthread_setconcurrency(threads);
}

void artsPthreadAffinity(unsigned int cpuCoreId) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(cpuCoreId, &cpuset);
    if(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
        PRINTF("Failed to set affinity %u\n", cpuCoreId);  
    
//    if (!pthread_getaffinity_np(thread, sizeof (cpu_set_t), &cpuset)) {
//        for (int i = 0; i < CPU_SETSIZE; i++) {
//            if (CPU_ISSET(i, &cpuset)) {
//                printf("%d ", i);
//            }
//        }
//    }
//    printf("\n");
}

int artsCheckAffinity() {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();

    int count = 0;
    int res = -1;
    bool flag = false;
    if (!pthread_getaffinity_np(thread, sizeof (cpu_set_t), &cpuset)) {
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &cpuset)) {
                res = i;
                count++;
            }
        }
    }
    
    if(count == 1)
        return res;
    return -1;
}