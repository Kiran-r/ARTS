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

unsigned int mat_size;
unsigned int tile_size;
unsigned int numBlocks = 1;

artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;

void printMatrix(unsigned int rowSize, double * mat)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            printf("%5.2f ", mat[i*rowSize + j]);
        }
        printf("\n");
    }
}

void initMatrix(unsigned int rowSize, double * mat, bool identity, bool zero)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
        for(unsigned int j=0; j<rowSize; j++)
            if(zero)
                mat[i*rowSize + j] = 0;
            else if(identity)
                if(i==j)
                    mat[i*rowSize + j] = 1;
                else
                    mat[i*rowSize + j] = 0;
            else
                mat[i*rowSize + j] =rand()%10;
}

void copyBlock(unsigned int x, unsigned int y, unsigned int tileRowSize, double * tile, unsigned int rowSize, double * mat, bool toTile)
{
    unsigned int tileColumnSize = tileRowSize;
    
    unsigned int xOffset = tileRowSize    * y;
    unsigned int yOffset = tileColumnSize * x;
    
    if(toTile)
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&tile[ i * tileRowSize ], &mat[ (i + yOffset) * rowSize + xOffset ], tileRowSize * sizeof(double));
    }
    else
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&mat[ (i + yOffset) * rowSize + xOffset ], &tile[ i * tileRowSize ], tileRowSize * sizeof(double));
    }

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
    artsReserveGuidRoute(edtType, rank);
    //visit right rank
    reserveEdtGuids(allGuids, right(index), edtType);
    return rank;
}

// binaryReductionTree_t * initBinaryReductionTree(unsigned int numLeaves, artsType_t dbType, artsType_t edtType)
// {
//     binaryReductionTree_t * tree = (binaryReductionTree_t*) artsCalloc(sizeof(binaryReductionTree_t));
//     tree->numLeaves = numLeaves;
//     tree->totalNodes = 2 * numLeaves - 1;
//     tree->interiorNodes = tree->totalNodes - tree->numLeaves;

//     artsGuid_t * allGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * tree->totalNodes);

//     for(unsigned int i=0; i<tree->numLeaves; i++)
//         allGuids[tree->interiorNodes + i] = artsReserveGuidRoute(dbType, i % artsGetTotalNodes());

//     reserveEdtGuids(allGuids, 0, edtType);

//     for(unsigned int i=0; i<tree->totalNodes; i++)
//         PRINTF("i: %u rank: %u type: %u\n", i, artsGuidGetRank(allGuids[i]), artsGuidGetType(allGuids[i]));

//     tree->redEdtGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * tree->interiorNodes);
//     for(unsigned int i=0; i<tree->interiorNodes; i++)
//     {
//         tree->redEdtGuids[i] = allGuids[i];
//         if(artsIsGuidLocal(tree->redEdtGuids[i]))
//             artsEdtCreateGpuPTWithGuid(functPtr, tree->redEdtGuids[i], )
//     }
    
//     tree->redDbGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * tree->numLeaves);
//     for(unsigned int i=0; i<tree->numLeaves; i++)
//         tree->redDbGuids[i] = allGuids[tree->interiorNodes + i] = artsReserveGuidRoute(dbType, i % artsGetTotalNodes());

//     artsFree(allGuids);
//     return tree;
// }

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    // initBinaryReductionTree(10, ARTS_DB_GPU_WRITE, ARTS_GPU_EDT);
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    artsShutdown();
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

/*
// void artsMoveLocalDBToGpu(artsGuid_t dbGuid)
// {
//     artsGpu_t artsGpu = artsGpus[index];
//     struct artsDb * dbTemp = artsRouteTableLookupItem(depv[i].guid);
//     void * dataPtr = artsGpuRouteTableLookupDb(depv[i].guid, artsGpu->device);
//     struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
//     size_t size = db->header.size - sizeof(struct artsDb);
//     if (!dataPtr)
//     {
//         //Actually allocate space
//         CHECKCORRECT(cudaMalloc(&dataPtr, size));
//         void * newDataPtr = artsGpuRouteTableAddItemRace(dataPtr, size, depv[i].guid, artsGpu->device);
//         if(newDataPtr == dataPtr) //We won, so move data
//         {
//             DPRINTF("Adding %lu %u id: %d\n", depv[i].guid, size, artsGpu->device);
//             CHECKCORRECT(cudaMemcpyAsync(dataPtr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsGpu->stream));
//             artsAtomicAdd(&misses, 1U);
//         }
//         else //Someone beat us to creating the data... So we must free
//         {
//             //Bug here...
//             cudaFree(dataPtr);
//             dataPtr = newDataPtr;
//             artsAtomicAdd(&falseMisses, 1U);
//         }
//     }
//     else
//     {
//         artsAtomicAddSizet(&artsGpu->availGlobalMem, size);
//         artsAtomicAdd(&hits, 1U);
//     }
// }

void internalPutInDb(void * ptr, artsGuid_t dbGuid, unsigned int offset, unsigned int size)
{
    unsigned int rank = artsGuidGetRank(dbGuid);
    if(rank==artsGlobalRankId)
    {
        struct artsDb * db = artsRouteTableLookupItem(dbGuid);
        if(db)
        {
            void * data = (void*)(((char*) (db+1)) + offset);
            // memcpy(data, ptr, size);
            CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost, artsLocalStream));
        }
        else
        {
            void * cpyPtr = artsMalloc(size);
            // memcpy(cpyPtr, ptr, size);
            CHECKCORRECT(cudaMemcpyAsync(data, ptr, size, cudaMemcpyDeviceToHost, artsLocalStream));
            artsOutOfOrderPutInDb(cpyPtr, NULL_GUID, dbGuid, 0, offset, size, NULL_GUID);
        }
    }

    for(unsigned int i=0; i<depc; i++)
    {
        if(depv[i].ptr && depv[i].mode == ARTS_DB_GPU_WRITE)
        {
            struct artsDb * db = (struct artsDb *) depv[i].ptr - 1;
            size_t size = (size_t) (db->header.size - sizeof(struct artsDb));
            CHECKCORRECT(cudaMemcpyAsync(depv[i].ptr, hostDepv[i].ptr, size, cudaMemcpyDeviceToHost, artsGpu->stream));
        }
    }
}

void artsPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size)
{
    ARTSEDTCOUNTERTIMERSTART(putDbCounter);
    unsigned int rank = artsGuidGetRank(dbGuid);
    artsGuid_t epochGuid = artsGetCurrentEpochGuid();
    DPRINTF("EPOCH %lu\n", epochGuid);
    incrementActiveEpoch(epochGuid);
    globalShutdownGuidIncActive();
    internalPutInDb(ptr, edtGuid, dbGuid, slot, offset, size, epochGuid, rank);
    ARTSEDTCOUNTERTIMERENDINCREMENT(putDbCounter);
}*/