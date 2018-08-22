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

u64 localTriangleCount = 0;
u64 time = 0;

u64 blockSize = 0;
u64 numBlocks = 0;

//Only support up to 64 nodes
unsigned int checkAndSet(u64 * mask, unsigned int index) {
    u64 bit = 1 << index;
    if(((*mask) & bit) == 0) {
        (*mask)|=bit;
        return 1;
    }
    return 0;
}

artsGuid_t finalReduce(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
    u64 count = 0;
    for (unsigned int i = 0; i < depc; i++) {
        count += (u64)depv[i].guid;
    }
    time = artsGetTimeStamp() - time;
    PRINTF("Triangle Count: %lu Time: %lu\n", count, time);
    artsShutdown();
}

artsGuid_t localReduce(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    artsSignalEdtValue(finalReduceGuid, artsGetCurrentNode(), localTriangleCount);
}

artsGuid_t startReduce(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
//    PRINTF("Local Count: %lu Signal: %lu\n", localTriangleCount, finalEdtGuid);
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
        artsEdtCreateDep(localReduce, i, 0, NULL, 0, false);
    }
}

uint64_t lowerBound(vertex value, u64 start, u64 end, vertex * edges) {
    while ((start < end) && (edges[start] < value))
        start++;
    return start;
}

uint64_t upperBound(vertex value, u64 start, u64 end, vertex * edges) {
    while ((start < end) && (value < edges[end - 1]))
        end--;
    return end;
}

u64 count_triangles(vertex * a, u64 a_start, u64 a_end, vertex * b, u64 b_start, u64 b_end) {
    u64 count = 0;
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

u64 processBlock(u64 index) {
    vertex * neighbors = NULL;
    u64 neighborCount = 0;
    u64 localCount = 0;
    
    u64 iStart = index*blockSize;
    u64 iEnd   = (index+1 == numBlocks) ? nodeEnd(artsGetCurrentNode(), &distribution) : iStart + blockSize;
    
    for (vertex i=iStart; i<iEnd; i++) {
        
        getNeighbors(&graph, i, &neighbors, &neighborCount);
        
        u64 firstPred = lowerBound(i, 0, neighborCount, neighbors);
        u64 lastPred = neighborCount;
        
        for (u64 nextPred = firstPred + 1; nextPred < lastPred; nextPred++) {
            vertex j = neighbors[nextPred];
            unsigned int owner = getOwner(j, &distribution);
            if (getOwner(j, &distribution) == artsGetCurrentNode()) {
                vertex * jNeighbors = NULL;
                u64 jNeighborCount = 0;
                getNeighbors(&graph, j, &jNeighbors, &jNeighborCount);
                u64 firstSucc = lowerBound(i, 0, jNeighborCount, jNeighbors);
                u64 lastSucc = upperBound(j, 0, jNeighborCount, jNeighbors);
                localCount += count_triangles(neighbors, firstPred, nextPred, jNeighbors, firstSucc, lastSucc);
            }
        }
    }
    return localCount;
}

artsGuid_t visitNode(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
    u64 localCount = 0;
    u64 index = paramv[0];
    
    localCount += processBlock(index);
    u64 nextIndex = (numBlocks - 1) - index;
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
    
    u64 size = end - start;
    blockSize = size / (artsGetTotalWorkers() * 32 * 2);
    numBlocks = size / blockSize;
    if(size % blockSize)
        numBlocks++;
    
    u64 half = numBlocks / 2;
    if(numBlocks % 2)
        half++;
    
    for (u64 index = 0; index < half; index++) {
        if(index % artsGetTotalWorkers() == workerId) {
            artsEdtCreate(visitNode, nodeId, 1, &index, 0);
        }
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
