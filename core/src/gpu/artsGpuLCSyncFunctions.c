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

// #define DPRINTF(...)
#define DPRINTF(...) PRINTF(__VA_ARGS__)

volatile unsigned int myLock = 0;

unsigned int versionLock(unsigned int * version)
{
    artsLock(&myLock);
    unsigned int local = *version;
    // while(1)
    // {
    //     if(local != (unsigned int)-1)
    //     {
    //         if(artsAtomicCswap(version, local, (unsigned int)-1) == local)
    //             break;
    //     }
    //     local = *version;
    // }
    // PRINTF("LOCK %u\n", local);
    return local;
}

void versionUnlock(unsigned int * version, unsigned int newVersion)
{
    artsUnlock(&myLock);
    // PRINTF("UNLOCK\n");
    // if(artsAtomicCswap(version, (unsigned int)-1, newVersion) == (unsigned int)-1)
    //     PRINTF("UNLOCK\n");
    // else
    //     PRINTF("FAILED UNLOCK\n");
    // *version = newVersion;
}

inline void artsPrintDbMetaData(artsLCMeta_t * db) 
{ 
    DPRINTF("guid: %lu ptr: %p dataSize: %lu hostVersion: %u gpuVersion: %u gpuTimeStamp: %u gpu: %d\n",  
        db->guid,
        db->data, 
        db->dataSize,
        *db->hostVersion,
        db->gpuVersion, 
        db->gpuTimeStamp, 
        db->gpu);
}

unsigned int artsGetLatestGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    artsPrintDbMetaData(host);
    artsPrintDbMetaData(dev);
    unsigned int ret = 0;
    unsigned int hostVersion = versionLock(host->hostVersion);
    // if(dev->hostVersion + dev->gpuVersion >= host->hostVersion + host->gpuVersion)
    {
        // if(dev->gpuTimeStamp > host->gpuTimeStamp)
        {
            DPRINTF("gpu: %d %u\n", dev->gpu, host->dataSize);
            // memcpy(host->data, dev->data, host->dataSize);
            host->gpuVersion = dev->gpuVersion;
            host->gpuTimeStamp = dev->gpuTimeStamp;
            host->gpu = dev->gpu;
            // ret = *dev->hostVersion + dev->gpuVersion;
        }
        unsigned int * p = (unsigned int*) host->data;
        unsigned int * p2 = (unsigned int*) dev->data;
        for(unsigned int i=0; i<host->dataSize/sizeof(unsigned int); i++)
        {
            p[i] = p2[i];
            printf("%u:%u ", p[i], p2[i]);
        }
        printf("\n");
    }
    versionUnlock(host->hostVersion, hostVersion);
    return 0;
}

unsigned int artsGetRandomGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    artsPrintDbMetaData(host);
    artsPrintDbMetaData(dev);
    unsigned int ret = 0;
    bool firstFlag = (host->gpu == -1);
    bool randomFlag = (artsThreadSafeRandom() % 2 == 1);
    DPRINTF("Random: %u %d\n", randomFlag, dev->gpu);
    if(firstFlag || randomFlag)
    {
        DPRINTF("gpu: %d\n", dev->gpu);
        memcpy(host->data, dev->data, host->dataSize);
        host->gpuVersion = dev->gpuVersion;
        host->gpuTimeStamp = dev->gpuTimeStamp;
        host->gpu = dev->gpu;
        // if(!firstFlag && randomFlag)
            // artsGpuInvalidateRouteTables(host->guid, (unsigned int) -1);
        ret = *dev->hostVersion + dev->gpuVersion;
    }
    return ret;
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