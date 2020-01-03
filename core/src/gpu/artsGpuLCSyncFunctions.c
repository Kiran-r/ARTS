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