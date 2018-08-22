#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "artsRT.h"
#include "artsGraph.h"
#include "artsTerminationDetection.h"
#include "artsAtomics.h"

arts_block_dist_t distribution;
csr_graph graph;

artsGuid_t epochGuid    = NULL_GUID;
artsGuid_t startReduceGuid = NULL_GUID;
artsGuid_t finalReduceGuid = NULL_GUID;

uint64_t localTriangleCount = 0;
uint64_t time = 0;

uint64_t blockSize = 0;
uint64_t numBlocks = 0;

//Only support up to 64 nodes
unsigned int checkAndSet(uint64_t * mask, unsigned int index) {
    uint64_t bit = 1 << index;
    if(((*mask) & bit) == 0) {
        (*mask)|=bit;
        return 1;
    }
    return 0;
}

void finalReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t count = 0;
    for (unsigned int i = 0; i < depc; i++) {
        count += (uint64_t)depv[i].guid;
    }
    time = artsGetTimeStamp() - time;
    PRINTF("Triangle Count: %lu Time: %lu\n", count, time);
    artsShutdown();
}

void localReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    artsSignalEdtValue(finalReduceGuid, artsGetCurrentNode(), localTriangleCount);
}

void startReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
        artsEdtCreateDep(localReduce, i, 0, NULL, 0, false);
    }
}

uint64_t lowerBound(vertex value, uint64_t start, uint64_t end, vertex * edges) {
    while ((start < end) && (edges[start] < value))
        start++;
    return start;
}

uint64_t upperBound(vertex value, uint64_t start, uint64_t end, vertex * edges) {
    while ((start < end) && (value < edges[end - 1]))
        end--;
    return end;
}

uint64_t count_triangles(vertex * a, uint64_t a_start, uint64_t a_end, vertex * b, uint64_t b_start, uint64_t b_end) {
    uint64_t count = 0;
    while ((a_start < a_end) && (b_start < b_end)) {
        if (a[a_start] < b[b_start])
            a_start++;
        else if (a[a_start] > b[b_start])
            b_start++;
        else {
            count++;
            a_start++;
            b_start++;
        }
    }
    return count;
}

uint64_t processBlock(uint64_t index) {
    vertex * neighbors = NULL;
    uint64_t neighborCount = 0;
    uint64_t localCount = 0;
    
    uint64_t iStart = index*blockSize;
    uint64_t iEnd   = (index+1 == numBlocks) ? nodeEnd(artsGetCurrentNode(), &distribution) : iStart + blockSize;
    
    for (vertex i=iStart; i<iEnd; i++) {
        
        getNeighbors(&graph, i, &neighbors, &neighborCount);
        
        uint64_t firstPred = lowerBound(i, 0, neighborCount, neighbors);
        uint64_t lastPred = neighborCount;
        
        for (uint64_t nextPred = firstPred + 1; nextPred < lastPred; nextPred++) {
            vertex j = neighbors[nextPred];
            unsigned int owner = getOwner(j, &distribution);
            if (getOwner(j, &distribution) == artsGetCurrentNode()) {
                vertex * jNeighbors = NULL;
                uint64_t jNeighborCount = 0;
                getNeighbors(&graph, j, &jNeighbors, &jNeighborCount);
                uint64_t firstSucc = lowerBound(i, 0, jNeighborCount, jNeighbors);
                uint64_t lastSucc = upperBound(j, 0, jNeighborCount, jNeighbors);
                localCount += count_triangles(neighbors, firstPred, nextPred, jNeighbors, firstSucc, lastSucc);
            }
        }
    }
    return localCount;
}

void visitNode(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t localCount = 0;
    uint64_t index = paramv[0];
    
    localCount += processBlock(index);
    uint64_t nextIndex = (numBlocks - 1) - index;
    if(nextIndex != index) {
        localCount += processBlock(nextIndex);
    }
    artsAtomicAddU64(&localTriangleCount, localCount);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
    initBlockDistributionWithCmdLineArgs(&distribution, argc, argv);
    loadGraphUsingCmdLineArgs(&graph, &distribution, argc, argv);

    startReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    finalReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    epochGuid       = artsInitializeEpoch(0, startReduceGuid, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {
    if(!nodeId && !workerId) {
        time = artsGetTimeStamp();
        artsEdtCreateWithGuid(startReduce, startReduceGuid, 0, NULL, 1);
        artsEdtCreateWithGuid(finalReduce, finalReduceGuid, 0, NULL, artsGetTotalNodes());
    }
    
    artsStartEpoch(epochGuid);
    vertex start = nodeStart(nodeId, &distribution);
    vertex end   = nodeEnd(nodeId, &distribution);
    
    uint64_t size = end - start;
    blockSize = size / (artsGetTotalWorkers() * 32 * 2);
    numBlocks = size / blockSize;
    if(size % blockSize)
        numBlocks++;
    
    uint64_t half = numBlocks / 2;
    if(numBlocks % 2)
        half++;
    
    for (uint64_t index = 0; index < half; index++) {
        if(index % artsGetTotalWorkers() == workerId) {
            artsEdtCreate(visitNode, nodeId, 1, &index, 0);
        }
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
