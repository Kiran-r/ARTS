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
#define TILE 16

uint64_t start = 0;

unsigned int numBlocks = 1;
artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;

void printMatrix(unsigned int rowSize, float * mat)
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

void initMatrix(unsigned int rowSize, float * mat, bool identity, bool zero)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            if(zero)
                mat[i*rowSize + j] = 0;
            else if(identity)
            {
                if(i==j)
                    mat[i*rowSize + j] = 1;
                else
                    mat[i*rowSize + j] = 0;
            }
            else
                mat[i*rowSize + j] = i * rowSize + j;
        }
    }
}

void copyBlock(unsigned int x, unsigned int y, unsigned int tileRowSize, float * tile, unsigned int rowSize, float * mat, bool toTile)
{
    unsigned int tileColumnSize = tileRowSize;
    
    unsigned int xOffset = tileRowSize    * x;
    unsigned int yOffset = tileColumnSize * y;
    
    if(toTile)
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&tile[ i * tileRowSize ], &mat[ (i + yOffset) * rowSize + xOffset ], tileRowSize * sizeof(float));
    }
    else
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&mat[ (i + yOffset) * rowSize + xOffset ], &tile[ i * tileRowSize ], tileRowSize * sizeof(float));
    }

}

__global__ void mmKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    const int blk = (int) paramv[0];
    float *A = (float *) depv[0].ptr; 
    float *B = (float *) depv[1].ptr;
    float *C = (float *) depv[2].ptr;
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    float sum = 0;
    for(unsigned int k=0; k<blk; k++)
        sum+=A[row * blk + k] * B[k * blk + col];
    C[row * blk + col] = sum;
}

void mmKernelCPU(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = (artsGuid_t) paramv[1];
    unsigned int k = (unsigned int) paramv[2];
    artsGuid_t cTileGuid = (artsGuid_t) paramv[3];
    const int blk = (int) paramv[0];
    float *A = (float *) depv[0].ptr; 
    float *B = (float *) depv[1].ptr;
    float *C = (float *) depv[2].ptr;
    
    for(unsigned int i=0; i<blk; i++)
    {
        //rows of B
        for(unsigned int j=0; j<blk; j++)
        {
            //rows of A and columns of B
            for(unsigned int k=0; k<blk; k++)
            {
                C[i * blk + j] += A[i * blk + k] * B[k * blk + j];
            }
        }
    }
    artsSignalEdt(toSignal, k, cTileGuid);
}

void multiplyMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0];
    
    unsigned int rowSize    = TILE;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    float * aMat = (float*) depv[0].ptr;
    float * bMat = (float*) depv[1].ptr;
    
    float * aTile = NULL;
    float * bTile = NULL;
    float * cTile = NULL;
    
    artsGuid_t aTileGuid = artsDbCreate((void**) &aTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    artsGuid_t bTileGuid = artsDbCreate((void**) &bTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    
    copyBlock(i, k, TILE, aTile, MATSIZE, aMat, true);
    copyBlock(k, j, TILE, bTile, MATSIZE, bMat, true);
    initMatrix(rowSize, cTile, false, true);
    
#ifdef GPUMM
    dim3 threads(TILE, TILE);
    dim3 grid(1, 1);
    
    uint64_t args[] = {TILE};
    artsGuid_t    mulGpuGuid = artsEdtCreateGpu(mmKernel, artsGetCurrentNode(), 1, args, 3, threads, grid, toSignal, k, cTileGuid);
    artsSignalEdt(mulGpuGuid, 0, aTileGuid);
    artsSignalEdt(mulGpuGuid, 1, bTileGuid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
#else
    uint64_t args[] = {TILE, toSignal, k, cTileGuid};
    artsGuid_t    mulGpuGuid = artsEdtCreate(mmKernelCPU, artsGetCurrentNode(), 4, args, 3);
    artsSignalEdt(mulGpuGuid, 0, aTileGuid);
    artsSignalEdt(mulGpuGuid, 1, bTileGuid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
#endif
}

void sumMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t doneGuid = paramv[0];
    
    unsigned int rowSize    = TILE;
    unsigned int columnSize = TILE;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    
    float * cTile;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    initMatrix(rowSize, cTile, false, true);
    
    for(unsigned int i=0; i<depc; i++)
    {
        float * toAdd = (float*) depv[i].ptr;
        for(unsigned int j=0; j<columnSize; j++)
        {
            for(unsigned int k=0; k<rowSize; k++)
            {
                cTile[j * rowSize + k] += toAdd[j * rowSize + k];
            }
        }
    }
//    printMatrix(TILE, cTile);
    artsSignalEdt(doneGuid, 1+ (i * numBlocks + j), cTileGuid);
}

void finishBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    float * cMat  = (float*) depv[0].ptr;
    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            float * cTile = (float*) depv[1 + i * numBlocks + j].ptr;
            copyBlock(i, j, TILE, cTile, MATSIZE, cMat, false);
        }
    }
    uint64_t time = artsGetTimeStamp() - start;
    
    for(unsigned int i=0; i<MATSIZE*MATSIZE; i++)
    {
        if(cMat[i] != i)
            PRINTF("Failed\n");
    }
//    printMatrix(MATSIZE, cMat);
    
    PRINTF("DONE %lu\n", time);
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numBlocks = MATSIZE / TILE;
    doneGuid = artsReserveGuidRoute(ARTS_EDT,     0);
    aMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    bMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    cMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    
    if(!nodeId)
    {
        float * aMat = (float*) artsDbCreateWithGuid(aMatGuid, MATSIZE * MATSIZE * sizeof(float));
        float * bMat = (float*) artsDbCreateWithGuid(bMatGuid, MATSIZE * MATSIZE * sizeof(float));
        float * cMat = (float*) artsDbCreateWithGuid(cMatGuid, MATSIZE * MATSIZE * sizeof(float));
        
        initMatrix(MATSIZE, aMat, false, false);
        initMatrix(MATSIZE, bMat,  true, false);
        initMatrix(MATSIZE, cMat, false, true);
        
//        PRINTF("A MATRIX\n");
//        printMatrix(MATSIZE, aMat);
//        PRINTF("B MATRIX\n");
//        printMatrix(MATSIZE, bMat);
//        PRINTF("C MATRIX\n");
//        printMatrix(MATSIZE, cMat);
        PRINTF("Starting\n");
    }
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    unsigned int totalThreads = artsGetTotalNodes() * artsGetTotalWorkers();
    unsigned int globalThreadId = nodeId * artsGetTotalWorkers() + workerId;
    
    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            if((i * numBlocks + j) % totalThreads == globalThreadId)
            {
                uint64_t sumArgs[] = {doneGuid, i, j};
                artsGuid_t sumGuid = artsEdtCreate(sumMM, nodeId, 3, sumArgs, numBlocks);
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {sumGuid, i, j, k};
                    artsGuid_t mulGuid = artsEdtCreate(multiplyMM, nodeId, 4, args, 2);
                    artsSignalEdt(mulGuid, 0, aMatGuid);
                    artsSignalEdt(mulGuid, 1, bMatGuid);
                }
            }
        }
    }
    
    if(!nodeId && !workerId)
    {
        artsEdtCreateWithGuid(finishBlockMM, doneGuid, 0, NULL, 1 + numBlocks * numBlocks);
        artsSignalEdt(doneGuid, 0, cMatGuid);
        start = artsGetTimeStamp();
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
