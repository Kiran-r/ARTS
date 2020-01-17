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
#include "arts.h"
#include "artsGpuRuntime.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#include <cuda_runtime.h>
#include <assert.h>

#define ARRAYSIZE 65536
#define TILESIZE 8192
#define VERIFY 1
#define SMTILE 32 // Hardcoded for Volta
#define NUMBINS 10 // Make it a variable

uint64_t start = 0;

int inputArraySize;
int tileSize;
unsigned int numBlocks = 1;

artsGuid_t inputArrayGuid = NULL_GUID;
artsGuid_t histoGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;
artsGuid_t finalSumGuid = NULL_GUID;

int * inputArray = NULL;
int * finalHistogram = NULL;

artsGuidRange * inputTileGuids = NULL;
artsGuidRange * partialHistoGuids = NULL;

__global__ void privateHistogram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    const int numElements = (int) paramv[0];
    const int numBins = (int) paramv[1];
    int * tile = (int *) depv[0].ptr;
    int * localHisto = (int *) depv[1].ptr;

    // Compute histograms in every GPU
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x; //32
    int id = threadIdx.x;
    __shared__ unsigned int sbins[NUMBINS];

    // for (int i = id; i < num_bins; i += step)
    //     if (i < num_bins)
    //         sbins[i] = 0;
    // __syncthreads();
    step = blockDim.x * gridDim.x;// 0-8192 /32 => 0-31, 32-63... 
    for (int i = index; i < numElements; i += step)
        if (i < numElements)
            atomicAdd(&sbins[tile[i]], 1);
    __syncthreads();
    step = blockDim.x;
    for (int i = id; i < numBins; i += step)
    {
        if (i < numBins)
            atomicAdd(&localHisto[i], sbins[i]);
    }
}

__global__ void ReduceHistogram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    // Reduce histograms from all GPUs into one.
    const int numBins = (int) paramv[0];
    const int numLocalHistograms = depc - 1;
    int * finalHisto = (int *) depv[0].ptr;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numBins)
    {
        for (int i = 0; i < numLocalHistograms; i++)
        {
            int * localHisto = (int *) depv[1+i].ptr;
            finalHisto[index] += localHisto[index];
        }
    }
    assert(1);
}

void finishHistogram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t time = artsGetTimeStamp() - start;
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    if (argc == 1)
    {
        inputArraySize = ARRAYSIZE;
        tileSize = TILESIZE;
    } else if (argc == 2)
    {
        inputArraySize = atoi(argv[1]);
        tileSize = TILESIZE;
    } else
    {
        inputArraySize = atoi(argv[1]);
        tileSize = atoi(argv[2]);
    }
    numBlocks = (inputArraySize+tileSize-1) / tileSize; // TODO: Fix if inputArraySize is < tileSize
    doneGuid = artsReserveGuidRoute(ARTS_EDT,            0);
    finalSumGuid = artsReserveGuidRoute(ARTS_GPU_EDT,    0);
    inputArrayGuid = artsReserveGuidRoute(ARTS_DB_READ,  0);
    histoGuid = artsReserveGuidRoute(ARTS_DB_WRITE,      0);

    inputTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_READ, numBlocks, 0);
    partialHistoGuids = artsNewGuidRangeNode(ARTS_DB_GPU_WRITE, numBlocks, 0);
    
    if (!nodeId)
    {
        inputArray = (int *) artsDbCreateWithGuid (inputArrayGuid, inputArraySize * sizeof(int));
        finalHistogram = (int *) artsDbCreateWithGuid (histoGuid, NUMBINS * sizeof(int));

        for (unsigned int elem = 0; elem < inputArraySize; elem++)
            inputArray[elem] = rand() % NUMBINS;
        
        for (unsigned int elem = 0; elem < NUMBINS; elem++)
            finalHistogram[elem] = 0;
        
        PRINTF("Starting...\n");
    }
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    unsigned int totalThreads = artsGetTotalNodes() * artsGetTotalWorkers();
    unsigned int globalThreadId = nodeId * artsGetTotalWorkers() + workerId;

    dim3 threads (SMTILE);
    dim3 grid((tileSize+SMTILE-1)/SMTILE);
    // dim3 grid(((tileSize*tileSize)+(SMTILE*SMTILE)-1)/(SMTILE*SMTILE), 1);

    if (!nodeId && !workerId)
    {
        for (unsigned int tile=0; tile<numBlocks; tile++)
        {
            artsGuid_t inputTileGuid = artsGetGuid(inputTileGuids, tile);
            artsGuid_t partialHistoGuid = artsGetGuid(partialHistoGuids, tile);
            int * inputTile = (int *) artsDbCreateWithGuid(inputTileGuid, sizeof(int) * tileSize);
            memcpy(inputTile, &inputArray[ tile * tileSize ], tileSize * sizeof(int));
            int * partialHisto = (int *) artsDbCreateWithGuid(partialHistoGuid, sizeof(int) *  NUMBINS);
            memset(partialHisto, 0, tileSize * sizeof(int));
        }
        uint64_t sumArgs[] = {tileSize};
        artsEdtCreateWithGuid (finishHistogram, doneGuid, 0, NULL, 1);
        artsEdtCreateGpuWithGuid (ReduceHistogram, finalSumGuid, 1, sumArgs, numBlocks+1, grid, threads, doneGuid, 0, histoGuid);
        artsSignalEdt(finalSumGuid, 0, histoGuid);
    }

    if (!workerId)
    {
        for (unsigned int tile=0; tile<numBlocks; tile++)
        {
            if (tile % artsGetTotalNodes() == nodeId)
            {
                artsGuid_t partialHistoGuid = artsGetGuid(partialHistoGuids, tile);
                uint64_t args[] = {numBlocks, tile};
                artsGuid_t privHistoGuid = artsEdtCreateGpu(privateHistogram, nodeId, 2, args, 2, grid, threads, finalSumGuid, 1+tile, partialHistoGuid);
                artsSignalEdt(privHistoGuid, 0, artsGetGuid(inputTileGuids, tile));
                artsSignalEdt(privHistoGuid, 1, partialHistoGuid);
            }
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}