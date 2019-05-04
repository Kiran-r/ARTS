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
//#include "artsGpuRuntime.h"

#define MATSIZE 10
#define TILE 2

uint64_t start = 0;

unsigned int numBlocks = 1;
artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
    
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
    unsigned int columnSize      = rowSize;
    
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

void initBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0];
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    float * aTile;
    float * bTile;
    
    artsGuid_t aTileGuid = artsDbCreate((void**) &aTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    artsGuid_t bTileGuid = artsDbCreate((void**) &bTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    
    float * aMat = (float*) depv[0].ptr;
    float * bMat = (float*) depv[1].ptr;
    
    copyBlock(i, k, TILE, aTile, MATSIZE, aMat, true);
    copyBlock(k, j, TILE, bTile, MATSIZE, bMat, true);
    
    artsSignalEdt(toSignal, 0, aTileGuid);
    artsSignalEdt(toSignal, 1, bTileGuid);
}

void multiplyMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0];
    
    unsigned int rowSize    = TILE;
    unsigned int columnSize = TILE;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    float * aTile = (float*) depv[0].ptr;
    float * bTile = (float*) depv[1].ptr;
    float * cTile = NULL;
    
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);
    initMatrix(rowSize, cTile, false, true);
    
    //columns of A
    for(unsigned int i=0; i<columnSize; i++)
    {
        //rows of B
        for(unsigned int j=0; j<rowSize; j++)
        {
            //rows of A and columns of B
            for(unsigned int k=0; k<rowSize; k++)
            {
                cTile[i * rowSize + j] += aTile[i * rowSize + k] * bTile[k * rowSize + j];
            }
        }
    }
    artsSignalEdt(toSignal, k, cTileGuid);
}

void sumMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t doneGuid = paramv[0];
    
    unsigned int rowSize    = TILE;
    unsigned int columnSize = TILE;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    float * cTile;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU);

    for(unsigned int i=0; i<depc; i++)
    {
        float * toAdd = depv[i].ptr;
        for(unsigned int j=0; j<columnSize; j++)
        {
            for(unsigned int k=0; k<rowSize; k++)
            {
                cTile[j * rowSize + k] += toAdd[j * rowSize + k];
            }
        }
    }    
    artsSignalEdt(doneGuid, i * numBlocks + j, cTileGuid);
}

void finishBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0]; 
    float * cMat  = (float*) depv[0].ptr;
    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            float * cTile = (float*) depv[i * numBlocks + j].ptr;
            copyBlock(i, j, TILE, cTile, MATSIZE, cMat, false);
        }
    }
    uint64_t time = artsGetTimeStamp() - start;
    printMatrix(MATSIZE, cMat);
    PRINTF("DONE %lu\n", time);
    artsShutdown();
}

//extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numBlocks = MATSIZE / TILE;
    
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

//extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        artsGuid_t doneGuid = artsEdtCreate(finishBlockMM, 0, 0, NULL, 1 + numBlocks * numBlocks);
        artsSignalEdt(doneGuid, 0, cMatGuid);
        
        for(unsigned int i=0; i<numBlocks; i++)
        {
            for(unsigned int j=0; j<numBlocks; j++)
            {
                uint64_t sumArgs[] = {doneGuid, i, j};
                artsGuid_t sumGuid = artsEdtCreate(sumMM, 0, 3, sumArgs, numBlocks);
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {sumGuid, i, j, k};
                    artsGuid_t mulGuid = artsEdtCreate(multiplyMM, 0, 4, args, 2);

                    args[0] = mulGuid; 
                    artsGuid_t initGuid = artsEdtCreate(initBlockMM, 0, 4, args, 2);
                    artsSignalEdt(initGuid, 0, aMatGuid);
                    artsSignalEdt(initGuid, 1, bMatGuid);
                }
            }
        }
        start = artsGetTimeStamp();
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
