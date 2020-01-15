/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,* 
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#include <cuda_runtime.h>
#include "artsGpuLCSyncFunctions.h"
#include "artsDbFunctions.h"
#include "artsGlobals.h"
#include "artsGpuRouteTable.h"
#include "artsDebug.h"
#include "artsAtomics.h"

#define DPRINTF(...)
// #define DPRINTF(...) PRINTF(__VA_ARGS__)

//To use this lock the unlock must be an even number
unsigned int versionLock(volatile unsigned int * version)
{
    unsigned int local = *version;
    while(1)
    {
        if((local & 1) == 0)
        {
            if(artsAtomicCswap(version, local, (local*2)+1) == local)
                break;
        }
        local = *version;
    }
    return local;
}

bool tryVersionLock(volatile unsigned int * version, unsigned int * currentVersion)
{
    unsigned int local = *version;
    if((local & 1) == 0)
    {
        if(artsAtomicCswap(version, local, (local*2)+1) == local)
        {
            *currentVersion = local;
            return true;
        }
    }
    *currentVersion = (local-1)/2;
    return false;
}

void versionUnlock(volatile unsigned int * version)
{
    unsigned int local = ((*version-1)/2) + 2;
    *version = local;
}

void versionUnlockWithNew(volatile unsigned int * version, unsigned int newVersion)
{
    if((newVersion & 1) == 0)
        *version = newVersion;
    else
        PRINTF("Must unlock with a positive version\n");
}

void * makeLCShadowCopy(struct artsDb * db)
{
    unsigned int size = db->header.size;
    void * dest = (void*)(((char*)db) + size);
    struct artsDb * shadowCopy = (struct artsDb*) dest;

    unsigned int hostVersion = versionLock(&db->version);
    if(hostVersion == shadowCopy->version)
        memcpy(dest, (void*)db, size);
    versionUnlockWithNew(&db->version, hostVersion);
    return dest;
}

inline void artsPrintDbMetaData(artsLCMeta_t * db) 
{ 
    DPRINTF("guid: %lu ptr: %p dataSize: %lu hostVersion: %u gpuVersion: %u gpuTimeStamp: %u gpu: %d\n",  
        db->guid,
        db->data, 
        db->dataSize,
        *db->hostVersion,
        *db->hostTimeStamp,
        db->gpuVersion, 
        db->gpuTimeStamp, 
        db->gpu);
}

void artsMemcpyGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int hostVersion = versionLock(host->hostVersion);
    memcpy(host->data, dev->data, host->dataSize);
    *host->hostTimeStamp = dev->gpuTimeStamp;
    versionUnlock(host->hostVersion);
}

void artsGetLatestGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int hostVersion = versionLock(host->hostVersion);
    if(*host->hostTimeStamp < dev->gpuTimeStamp)
    {
            memcpy(host->data, dev->data, host->dataSize);
            host->gpuVersion = dev->gpuVersion;
            host->gpuTimeStamp = dev->gpuTimeStamp;
            *host->hostTimeStamp = dev->gpuTimeStamp;
            host->gpu = dev->gpu;
    }
    versionUnlock(host->hostVersion);
}

void artsGetRandomGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    bool firstFlag = (host->gpu == -1);
    bool randomFlag = ((artsThreadSafeRandom() & 1) == 0);
    if(firstFlag || randomFlag)
    {
        unsigned int currentVersion;
        if(tryVersionLock(host->hostVersion, &currentVersion))
        {
            memcpy(host->data, dev->data, host->dataSize);
            host->gpuVersion = dev->gpuVersion;
            host->gpuTimeStamp = dev->gpuTimeStamp;
            *host->hostTimeStamp = dev->gpuTimeStamp;
            host->gpu = dev->gpu;
            // if(!firstFlag && randomFlag)
                // artsGpuInvalidateRouteTables(host->guid, (unsigned int) -1);
        }
        
    }
}

void artsGetNonZerosUnsignedInt(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int numElem = host->dataSize/sizeof(unsigned int);
    unsigned int * dest = (unsigned int*) host->data;
    unsigned int * src = (unsigned int*) dev->data;
    // unsigned int hostVersion = versionLock(host->hostVersion);
    for(unsigned int i=0; i<numElem; i++)
    {
        DPRINTF("src: %u dest: %u\n", src[i], dest[i]);
        if(src[i])
            dest[i] = src[i];
    }
    // versionUnlock(host->hostVersion);
}

// void artsGetMinDb(artsLCMeta_t * host, artsLCMeta_t * dev)
// {
//     artsPrintDbMetaData(host);
//     artsPrintDbMetaData(dev)
    
//     unsigned int numElements = host->dataSize / sizeof(unsigned int);
//     for(unsigned int i=0; i<numElements; i++) 
//     {

//     }
// }

#define GPUGROUPSIZE 4
#define GPUNUMGROUP 2
int internalFindRoots(unsigned int local)
{
    unsigned int mask = 0;
    for(unsigned int j=0; j<GPUGROUPSIZE; j++)
    {
        unsigned int bit = 1 << j;
        mask |= bit;
    }

    unsigned int roots = -1;
    for(unsigned int i=0; i<GPUNUMGROUP; i++)
    {
        unsigned int tempLocal = local >> (i*GPUGROUPSIZE);
        unsigned int temp = mask & tempLocal;
        roots &= temp;
    }
    
    for(int i=0; i<GPUGROUPSIZE; i++)
    {
        if(roots & (1 << i))
            return i;
    }
    return -1;
}

void internalLaunchReduction(int root, int a, int b)
{
    if(root != a && root != b)
    {
        PRINTF("LC Reduction tree invalid root! %d %d %d\n", root, a, b);
       artsDebugGenerateSegFault(); 
    }
    if(a < 0 || b < 0 )
        return;
    PRINTF("A: %d B: %d -> Root: %d\n", a, b, root);
}

int internalGPUGroupReduction(int root, unsigned int start, unsigned int stop, unsigned int mask)
{
    // PRINTF("root: %u start: %u stop: %u\n", root, start, stop);
    int localRoot = -1;
    int gpuId[2] = {start, stop};

    if(stop - start > 1) //Recursive call
    {
        unsigned int middle = (1 + stop - start) / 2;
        gpuId[0] = internalGPUGroupReduction(root, start, start + middle - 1, mask);
        gpuId[1] = internalGPUGroupReduction(root, start + middle, stop, mask);
    }

    bool startFound = (gpuId[0] < 0) ? false : ((mask & (1 << gpuId[0])) != 0);
    bool stopFound =  (gpuId[1] < 0) ? false : ((mask & (1 << gpuId[1])) != 0);

    if(startFound && stopFound) //Both are in the mask
    {
        if(root == start || root == stop)
            localRoot = root;
        else
            localRoot = start; //This is the min
    }
    else if(startFound && !stopFound) //Only start is in the mask
    {
        gpuId[1] = -1;
        localRoot = startFound;
    }
    else if(!startFound && stopFound) //Only stop is in the mask
    {
        gpuId[0] = -1;
        localRoot = stopFound;
    }
    else //Neither start or stop is in the mask
    {
        gpuId[1] = -1;
        gpuId[0] = -1;
        // localRoot = -1;
    }
    
    internalLaunchReduction(localRoot, gpuId[0], gpuId[1]);
    return localRoot;
}

void artsSendTree(unsigned int mask)
{
    if(mask)
    {
        int root[GPUNUMGROUP];
        root[0] = internalFindRoots(mask);
        for(unsigned int i=0; i<GPUNUMGROUP; i++)
        {
            if(root[0] < 0)
                root[i] = -1; //i*GPUGROUPSIZE;
            else
                root[i] = root[0] + i*GPUGROUPSIZE;
            internalGPUGroupReduction(root[i], i * GPUGROUPSIZE, ((i+1) * GPUGROUPSIZE) - 1, mask);
        }
    }
}
