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

#define GPUMM 1
#define MATSIZE 1024
#define TILESIZE 16
#define VERIFY 1

uint64_t start = 0;

unsigned int matSize;
unsigned int tileSize;
unsigned int numBlocks = 1;

artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;

void shutdown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("SHUTDOWN\n");
    artsShutdown();
}

__global__ void sumMMKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    const unsigned int columnSize = (unsigned int) paramv[0];
    double * cTile = (double *) depv[0].ptr;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    for (unsigned int k=1; k<depc; ++k)
    {
        double* toAdd = (double*) depv[k].ptr;
        cTile[row * columnSize + col] += toAdd[row*columnSize+col];
    }
}

typedef struct {
    unsigned int numLeaves;
    unsigned int totalNodes;
    unsigned int interiorNodes;
    artsGuid_t * redDbGuids;
    artsGuid_t * redEdtGuids;
} binaryReductionTree_t;

unsigned int left(unsigned int i) { return 2*i + 1; }
unsigned int right(unsigned int i) { return 2*i + 2; }
unsigned int parent(unsigned int i) { return (i-1)/2; }

unsigned int reserveEdtGuids(artsGuid_t * allGuids, unsigned int index, artsType_t edtType)
{
    if(allGuids[index])
        return artsGuidGetRank(allGuids[index]);
    //left Rank    
    unsigned int rank = reserveEdtGuids(allGuids, left(index), edtType);
    //always reserve left rank
    allGuids[index] = artsReserveGuidRoute(edtType, rank);
    PRINTF("edt: %d -> %lu\n", index, allGuids[index]);
    //visit right rank
    reserveEdtGuids(allGuids, right(index), edtType);
    return rank;
}

binaryReductionTree_t * initBinaryReductionTree(unsigned int numLeaves, artsEdt_t funPtr, artsType_t dbType, artsType_t edtType, uint32_t paramc, uint64_t * paramv, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot)
{
    binaryReductionTree_t * tree = (binaryReductionTree_t*) artsCalloc(sizeof(binaryReductionTree_t));
    tree->numLeaves = numLeaves;
    tree->totalNodes = 2 * numLeaves - 1;
    tree->interiorNodes = tree->totalNodes - tree->numLeaves;

    //Create space for all the guids
    artsGuid_t * allGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * tree->totalNodes);
    tree->redDbGuids = &allGuids[tree->interiorNodes];
    tree->redEdtGuids = allGuids;

    //Reserves the db guids
    for(unsigned int i=0; i<tree->numLeaves; i++)
        allGuids[tree->interiorNodes + i] = artsReserveGuidRoute(dbType, i % artsGetTotalNodes());

    //Reserves the edt guids
    reserveEdtGuids(allGuids, 0, edtType);

    //Check all the guids
    for(unsigned int i=0; i<tree->totalNodes; i++)
        PRINTF("i: %u guid: %lu rank: %u type: %u\n", i, allGuids[i], artsGuidGetRank(allGuids[i]), artsGuidGetType(allGuids[i]));

    //Set up the signals
    for(unsigned int i=0; i<tree->interiorNodes; i++)
    {
        if(artsIsGuidLocal(tree->redEdtGuids[i]))
        {
            if(!i)
            {
                PRINTF("Last: %lu -> %lu slot: %u\n", tree->redEdtGuids[i], endGuid, slot);
                artsEdtCreateGpuPTWithGuid(funPtr, tree->redEdtGuids[i], paramc, paramv, 2, grid, block, endGuid, 0, slot);
            }
            else
            {
                int parentIndex = parent(i);
                bool isRight = right(parentIndex) == i;
                artsGuid_t toSignal = tree->redEdtGuids[parentIndex];
                int toSignalSlot = (isRight) ? 1 : 0;
                PRINTF("%lu -> %lu slot: %u parent: %d\n", tree->redEdtGuids[i], toSignal, toSignalSlot, parentIndex);
                artsEdtCreateGpuPTWithGuid(funPtr, tree->redEdtGuids[i], paramc, paramv, 2, grid, block, toSignal, 0, toSignalSlot);
            }
        }
    }

    return tree;
}

void fireBinaryReductionTree(binaryReductionTree_t * tree)
{
    //Signal the top edts
    for(unsigned int i=0; i<tree->numLeaves; i++)
    {
        int index = tree->interiorNodes + i;
        int parentIndex = parent(index);
        bool isRight = right(parentIndex) == i;
        artsGuid_t toSignal = tree->redEdtGuids[parentIndex];
        int toSignalSlot = (isRight) ? 1 : 0;
        artsSignalEdt(toSignal, toSignalSlot, tree->redDbGuids[i]);
    }
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    artsGuid_t doneGuid = artsReserveGuidRoute(ARTS_EDT, 0);

    uint64_t sumArgs[] = {tileSize};
    dim3 threads(tileSize, tileSize);
    dim3 grid(1, 1);
    PRINTF("doneGuid: %lu\n", doneGuid);
    initBinaryReductionTree(8, sumMMKernel, ARTS_DB_GPU_WRITE, ARTS_GPU_EDT, 1, sumArgs, grid, threads, doneGuid, 0);

}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
        artsEdtCreateWithGuid(shutdown, doneGuid, 0, NULL, 1);
    artsShutdown();
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
