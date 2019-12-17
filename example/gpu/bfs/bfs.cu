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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "arts.h"
#include "artsGraph.h"
#include "artsTerminationDetection.h"

#include "artsGpuRuntime.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include "graphUtil.h"

#define DPRINTF(...)
// #define DPRINTF(...) PRINTF(__VA_ARGS__)
#define TURNON(...) 
// #define TURNON(...) __VA_ARGS__
#define ROOT 1
#define PARTS 8
#define GPULISTLEN 2048*2048
#define MAXLEVEL (unsigned int) -1
#define SMTILE 32

uint64_t start = 0; //Timer
unsigned int ** devPtrRaw; //The pointers for our nextSearchFrontier on each gpu
artsGuid_t nextSearchFrontierAddrGuid; //db that holds guids for the nextSearchFrontiers (devPtrRaw)
unsigned int bounds[PARTS]; //This is the boundaries that make up each partition
arts_block_dist_t * distribution; //The graph distribution
csr_graph_t * graph; //Partitions of the graph
unsigned int ** visited; //This is the resulting parent list for each partition
artsGuid_t * visitedGuid; //This is the guid for each partition of the parent list
uint64_t level = 0; //The current level of search we are at


void launchBfs(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void launchSort(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);

void printDeviceList(thrust::device_ptr<unsigned int> devPtr, unsigned int size)
{
    PRINTF("FRONTIER SIZE: %u\n", size);
    for(unsigned int i=0; i<size; i++)
    {
        unsigned int temp = *(devPtr + i);
        printf("%u, ", temp);
    }
    printf("\n");
}

void printResult()
{
    for(unsigned int i=0; i<PARTS; i++)
    {
        unsigned int size = sizeof(unsigned int) * getBlockSizeForPartition(i, distribution);
        printf("%u: %u\n", i, size);
        for(unsigned int j=0; j<size; j++)
            printf("%u, ", visited[i][j]);
        printf("\n");
    }
}

__global__ void bfs(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int localLevel = (unsigned int) paramv[0];
    uint64_t gpuId = getGpuIndex(); //The current gpu we are on
    unsigned int ** addr = (unsigned int **)depv[0].ptr; //This is the devPtrRaw -> tells us where current frontier is on device
    unsigned int * local = addr[gpuId]; //We need the one corresponding to our gpu
    unsigned int * localFrontierCount = &local[GPULISTLEN];

    unsigned int currentFrontierSize = *((unsigned int*)depv[1].ptr);
    unsigned int * currentFrontier = ((unsigned int*)depv[1].ptr) + 1;
    csr_graph_t * localGraph = (csr_graph_t*) depv[2].ptr;
    unsigned int * localVisited = (unsigned int*)depv[3].ptr;

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < currentFrontierSize)
    {
        vertex_t v = currentFrontier[index];
        local_index_t vertexIndex = getLocalIndexGpu(v, localGraph);
        unsigned int oldLevel = localVisited[vertexIndex];
        bool success = false;
        while(localLevel < oldLevel)
        {
            success = (atomicCAS(&localVisited[vertexIndex], oldLevel, localLevel) == oldLevel);
            oldLevel = localVisited[vertexIndex];
        }

        if(success)
        {
            vertex_t* neighbors = NULL;
            uint64_t neighborCount = 0;
            getNeighborsGpu(localGraph, v, &neighbors, &neighborCount);
            if(neighborCount)
            {
                unsigned int frontierIndex = atomicAdd(localFrontierCount, (unsigned int)neighborCount);
                if(frontierIndex < GPULISTLEN)
                {
                    for (uint64_t i = 0; i < neighborCount; ++i) 
                    {
                        local[frontierIndex+i] = neighbors[i]; 
                    }
                }
            }
        }
    }
}

void thrustSort(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    DPRINTF("%s %u\n", __func__, level);
    artsGuid_t nextGuid = paramv[0]; //This can be the end if the frontier is empty
    unsigned int gpuIndex = paramv[1]; //gpuIndex
    artsGuid_t nextLaunchSortGuid = paramv[2]; // To let us launch next bfs-es
    unsigned int * rawPtr = devPtrRaw[gpuIndex]; //The corresponding dev pointer (frontier) to our gpu
    artsGuid_t edtGuidsToLaunchGuid = NULL_GUID;

    //Get frontier count
    thrust::device_ptr<unsigned int> devCounterPtr(rawPtr + GPULISTLEN);
    unsigned int newFrontierCount = *(devCounterPtr);
    if(newFrontierCount <= GPULISTLEN) //If it was bigger than the search frontier, we will exit in the launchBfs
    {
        //Sort the frontier
        thrust::device_ptr<unsigned int> devPtr(rawPtr);
        thrust::sort(devPtr, devPtr+newFrontierCount); //Do the sorting
        TURNON(printDeviceList(devPtr, newFrontierCount));

        //Remove duplicates
        newFrontierCount = thrust::unique(thrust::device, devPtr, devPtr+newFrontierCount) - devPtr;
        TURNON(printDeviceList(devPtr, newFrontierCount));

        //Reset frontier
        *(devCounterPtr) = 0;

        //Get the boundery of each partition
        unsigned int * upperIndexPerBound = (unsigned int*) artsCalloc(sizeof(unsigned int)*PARTS);
        for(unsigned int i=0; i<PARTS; i++)
            upperIndexPerBound[i] = thrust::upper_bound(thrust::device, devPtr, devPtr+newFrontierCount, bounds[i]) - devPtr;

        //Get the size of each partition
        unsigned int * sizePerBound = (unsigned int*) artsCalloc(sizeof(unsigned int)*PARTS);
        sizePerBound[0] = upperIndexPerBound[0];
        DPRINTF("Upper: %u Size: %u\n", bounds[0], sizePerBound[0]);
        for(unsigned int i=1; i<PARTS; i++)
        {
            sizePerBound[i] = upperIndexPerBound[i] - upperIndexPerBound[i-1];
            DPRINTF("Upper: %u Size: %u\n", bounds[i], sizePerBound[i]);
        }

        //TODO: Clear old dbs (previous frontiers)...

        artsGuid_t * edtGuidsToLaunch = NULL; //This will hold the new edt guids to launch
        edtGuidsToLaunchGuid = artsDbCreate((void**) &edtGuidsToLaunch, sizeof(artsGuid_t) * PARTS, ARTS_DB_READ);

        uint64_t bfsArgs[] = {level};
        unsigned tempIndex = 0;
        for(unsigned int i=0; i<PARTS; i++) 
        {
            if(sizePerBound[i])
            {
                unsigned int * newSearchFrontier = NULL; //This will hold a tile of the new frontier
                artsGuid_t newSearchFrontierGuid = artsDbCreate((void**) &newSearchFrontier, sizeof(unsigned int) * (sizePerBound[i] + 1), ARTS_DB_GPU_READ);
                *newSearchFrontier = sizePerBound[i];

                //Copy the data from the gpu to the host
                artsPutInDbFromGpu(thrust::raw_pointer_cast(devPtr) + tempIndex, newSearchFrontierGuid, sizeof(unsigned int), sizeof(unsigned int) * sizePerBound[i], false);
                tempIndex+=sizePerBound[i];

                dim3 threads(SMTILE, 1, 1);
                dim3 grid((sizePerBound[i] + SMTILE - 1) / SMTILE, 1, 1); //Ceiling

                //Create the new edt for each bfs
                edtGuidsToLaunch[i] = artsEdtCreateGpu(bfs, artsGetCurrentNode(), 1, bfsArgs, 4, grid, threads, nextLaunchSortGuid, 0, NULL_GUID);
                artsSignalEdt(edtGuidsToLaunch[i], 1, newSearchFrontierGuid);
                artsSignalEdt(edtGuidsToLaunch[i], 2, getGuidForPartitionDistr(distribution, i));
                artsSignalEdt(edtGuidsToLaunch[i], 3, visitedGuid[i]);
            }
            else
                edtGuidsToLaunch[i] = NULL_GUID;
        }
    }
    artsSignalEdt(nextGuid, gpuIndex, edtGuidsToLaunchGuid);
}

void launchBfs(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    DPRINTF("%s %u\n", __func__, level);
    if(level < MAXLEVEL) //This is a cut-off for debugging purposes
    {
        //Walk through and signal the new edts that we are ready to run!
        unsigned int numberOfNewBfs = 0;
        for(unsigned int i=0; i<depc; i++)
        {
            if(depv[i].guid)  //This makes sure the search frontier was big enough
            {
                artsGuid_t * guidToSignal = (artsGuid_t*) depv[i].ptr;
                for(unsigned int j=0; j<PARTS; j++)
                {
                    if(guidToSignal[j])
                    {
                        artsSignalEdt(guidToSignal[j], 0, nextSearchFrontierAddrGuid);
                        numberOfNewBfs++;
                    }
                }
            }
            else  //We are exiting bc the search frontier was too small...
                PRINTF("FRONTIER NOT BIG ENOUGH SHUTTING DOWN...\n");   
        }

        if(numberOfNewBfs) //We have more work so lets launch the next sync point
        {
            DPRINTF("NUMBEROFNEWBFS: %u\n", numberOfNewBfs);
            artsGuid_t nextLaunchSortGuid = (artsGuid_t) paramv[0];
            artsEdtCreateWithGuid(launchSort, nextLaunchSortGuid, 0, NULL, numberOfNewBfs);
            level++;
            return;
        }
        else //No more work, lets shut it down!
        {
            uint64_t time = artsGetTimeStamp() - start;
            PRINTF("Done. Total execution time: %f s\n", (double) time /1000000000.0 );
            TURNON(printResult());
        }
    }
    else
        PRINTF("LEVEL SHUT DOWN\n");
    artsShutdown();
}

void launchSort(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    DPRINTF("%s %u\n", __func__, level);
    level++;
    //Use out-of-order engine!!!
    artsGuid_t nextLaunchSortGuid = artsReserveGuidRoute(ARTS_EDT, artsGetCurrentNode());
    //Launch next sync for each of the gpu sorts
    artsGuid_t launchBfsGuid = artsEdtCreate(launchBfs, artsGetCurrentNode(), 1, (uint64_t*) &nextLaunchSortGuid, artsGetTotalGpus());

    dim3 threads(1, 1, 1);
    dim3 grid(1, 1, 1);
    for(uint64_t i=0; i<artsGetTotalGpus(); i++)
    {
        uint64_t args[] = {launchBfsGuid, i, nextLaunchSortGuid}; 
        artsGuid_t sortGuid = artsEdtCreateGpuLibDirect(thrustSort, artsGetCurrentNode(), i, 3, args, 0, grid, threads);
    }
}

void createFirstRound(artsGuid_t initialSearchFrontier, artsGuid_t graphGuid, artsGuid_t visitGuid)
{
    dim3 threads (1, 1, 1);
    dim3 grid (1, 1, 1);
    artsGuid_t launchSortGuid = artsEdtCreate(launchSort, artsGetCurrentNode(), 0, NULL, 1);
    uint64_t bfsArgs[] = {level};
    artsGuid_t bfsGuid = artsEdtCreateGpu(bfs, artsGetCurrentNode(), 1, bfsArgs, 4, grid, threads, launchSortGuid, 0, NULL_GUID);
    artsSignalEdt(bfsGuid, 0, nextSearchFrontierAddrGuid);
    artsSignalEdt(bfsGuid, 1, initialSearchFrontier);
    artsSignalEdt(bfsGuid, 2, graphGuid);
    artsSignalEdt(bfsGuid, 3, visitGuid);
    DPRINTF("LAUNCHING\n");
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    DPRINTF("%s\n", __func__);
    char * fileName = NULL;
    unsigned int numVerts = 0;
    unsigned int numEdges = 0;
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--file", argv[i]) == 0) {
	fileName = argv[i + 1];
      }
      if (strcmp("--numvertices", argv[i]) == 0) {
	sscanf(argv[i + 1], "%u", &numVerts);
      }
      if (strcmp("--numedges", argv[i]) == 0) {
	sscanf(argv[i + 1], "%u", &numEdges);
      }
    }
    // char * fileName = "/home/firo017/datasets/ca-HepTh.tsv";    // "/home/suet688/ca-HepTh.tsv";
    // char * fileName = "/home/firo017/datasets/ca-HepPh_adj.tsv";
    // unsigned int numVerts = 9877;
    // unsigned int numEdges = 51946;
    // unsigned int numVerts = 12008;
    // unsigned int numEdges = 118521;
    // if(argc == 4)
    // {
    //     fileName = argv[1];
    //     numVerts = atoi(argv[2]);
    //     numEdges = atoi(argv[3]);
    // }
    //Create graph partitions
    graph = (csr_graph_t*) artsCalloc(sizeof(csr_graph_t)* PARTS);
    distribution = initBlockDistributionBlock(numVerts, numEdges, PARTS, ARTS_DB_GPU_READ);
    loadGraphNoWeight(fileName, distribution, true, false);

    for(unsigned int i=0; i<PARTS; i++)
    {
        bounds[i] = partitionEndDistr(i, distribution);
        DPRINTF("Bounds[%u]: %lu guid: %lu\n", i, bounds[i], distribution->graphGuid[i]);
    }

    //Create visited array per partition
    visitedGuid = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * PARTS);
    visited = (unsigned int**)artsCalloc(sizeof(unsigned int*)*PARTS);
    for(unsigned int i=0; i<PARTS; i++)
    {
        unsigned int size = sizeof(unsigned int) * getBlockSizeForPartition(i, distribution);
        unsigned int rank = artsGuidGetRank(getGuidForPartitionDistr(distribution, i));
        visitedGuid[i] = artsReserveGuidRoute(ARTS_DB_GPU_WRITE, rank); //Put the visiter db on the same rank as the graph partition
        //If the partitionis on our node lets create the db and -1 it out
        if(rank == nodeId)
        {
            visited[i] = (unsigned int*)artsDbCreateWithGuid(visitedGuid[i], size);
            for(unsigned int j=0; j<size; j++)
                visited[i][j] = UINT32_MAX;
        }
    }

    //Create an array to hold the addresses of next search frontier for each gpu
    devPtrRaw = (unsigned int**) artsCalloc(sizeof(unsigned int*) * artsGetTotalGpus());
}

extern "C"
void initPerGpu(unsigned int nodeId, int devId, cudaStream_t * stream, int argc, char * argv)
{
    DPRINTF("%s\n", __func__);
    //Create the next search frontier for each gpu
    devPtrRaw[devId] = (unsigned int*) artsCudaMalloc(sizeof(unsigned int) * (GPULISTLEN+1));
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    DPRINTF("%s\n", __func__);
    if (!workerId) {
        //Lets create the db of to hold device address of the next search frontier so gpu kernels can use them 
        unsigned int ** addr;
        nextSearchFrontierAddrGuid = artsDbCreate((void**)&addr, sizeof(unsigned int*) * artsGetTotalGpus(), ARTS_DB_GPU_READ);
        for(uint64_t i=0; i<artsGetTotalGpus(); i++)
            addr[i] = devPtrRaw[i];

        //Create the first search frontier!
        unsigned int * firstSearchFrontier = NULL;
        artsGuid_t firstSearchFrontierGuid = artsDbCreate((void**) &firstSearchFrontier, 2*sizeof(unsigned int), ARTS_DB_GPU_READ);
        firstSearchFrontier[0] = 1;
        firstSearchFrontier[1] = ROOT;
        DPRINTF("ROOT: %u GRAPH GUID: %lu VISITED GUID: %lu\n", firstSearchFrontier[1], getGuidForVertexDistr(firstSearchFrontier[1], distribution), visitedGuid[getOwnerDistr(firstSearchFrontier[1], distribution)]);
        createFirstRound(firstSearchFrontierGuid, getGuidForVertexDistr(firstSearchFrontier[1], distribution), visitedGuid[getOwnerDistr(firstSearchFrontier[1], distribution)]);
        start = artsGetTimeStamp();
    }
}

extern "C"
void cleanPerGpu(unsigned int nodeId, int devId, cudaStream_t * stream)
{
    DPRINTF("%s\n", __func__);
    artsCudaFree(devPtrRaw[devId]);
}

int main(int argc, char** argv) {
    DPRINTF("%s\n", __func__);
    artsRT(argc, argv);
    return 0;
}
