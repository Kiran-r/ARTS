#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"
#include "hiveAtomics.h"

hive_block_dist_t distribution;
csr_graph graph;

hiveGuid_t epochGuid       = NULL_GUID;
hiveGuid_t startReduceGuid = NULL_GUID;
hiveGuid_t finalReduceGuid = NULL_GUID;

vertex distStart = 0;
vertex distEnd   = 0;
u64 blockSize    = 0;
u64 overSub      = 16;

u64 otherCount = 0;
u64 localTriangleCount = 0;
u64 time = 0;

u64 local = 0;
u64 remote = 0;
u64 incoming = 0;

hiveGuid_t finalReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);
hiveGuid_t localReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);
hiveGuid_t startReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);
hiveGuid_t visitVertex(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

//Only support up to 64 nodes
inline unsigned int checkAndSet(u64 * mask, unsigned int index) {
    u64 bit = 1 << index;
    if(((*mask) & bit) == 0) {
        (*mask)|=bit;
        return 1;
    }
    return 0;
}

hiveGuid_t finalReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    u64 count = 0;
    for (unsigned int i = 0; i < depc; i++) {
        count += (u64)depv[i].guid;
    }
    time = hiveGetTimeStamp() - time;
    PRINTF("Triangle Count: %lu Time: %lu\n", count, time);
    hiveShutdown();
}

hiveGuid_t localReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    PRINTF("Local: %lu Remote: %lu Incoming: %lu\n", local, remote, incoming);
    hiveSignalEdtValue(finalReduceGuid, hiveGetCurrentNode(), localTriangleCount+otherCount);
}

hiveGuid_t startReduce(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++) {
        hiveEdtCreateDep(localReduce, i, 0, NULL, 0, false);
    }
}

inline uint64_t lowerBound(vertex value, u64 start, u64 end, vertex * edges) {
    while ((start < end) && (edges[start] < value))
        start++;
    return start;
}

inline uint64_t upperBound(vertex value, u64 start, u64 end, vertex * edges) {
    while ((start < end) && (value < edges[end - 1]))
        end--;
    return end;
}

inline u64 countTriangles(vertex * a, u64 a_start, u64 a_end, vertex * b, u64 b_start, u64 b_end) {
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

inline u64 processVertex(vertex i, vertex * neighbors, u64 neighborCount, u64 * visitMask, u64 * procLocal, u64 * procRemote) {
    u64 localCount = 0;
    
    u64 firstPred = lowerBound(i, 0, neighborCount, neighbors);
    u64 lastPred = neighborCount;
    
    for (u64 nextPred = firstPred + 1; nextPred < lastPred; nextPred++) {
        vertex j = neighbors[nextPred];
        unsigned int owner = getOwner(j, &distribution);
        if (getOwner(j, &distribution) == hiveGetCurrentNode()) {
            vertex * jNeighbors = NULL;
            u64 jNeighborCount = 0;
            getNeighbors(&graph, j, &jNeighbors, &jNeighborCount);
            u64 firstSucc = lowerBound(i, 0, jNeighborCount, jNeighbors);
            u64 lastSucc = upperBound(j, 0, jNeighborCount, jNeighbors);
            localCount += countTriangles(neighbors, firstPred, nextPred, jNeighbors, firstSucc, lastSucc);
            (*procLocal)++;
        }
        else if(checkAndSet(visitMask, owner)) {
            u64 args[3];
            args[0] = i;
            args[1] = i;
            args[2] = neighborCount;
            hiveGuid_t guid = hiveEdtCreate(visitVertex, owner, 3, args, 1);
            hiveSignalEdtPtr(guid, 0, neighbors, sizeof(vertex) * neighborCount);
            (*procRemote)++;
        }
    }
    return localCount;
}

hiveGuid_t visitVertex(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    u64 localCount = 0;
    
    vertex start = paramv[0];
    vertex end   = paramv[1];
    
    vertex * neighbors = NULL;
    u64 neighborCount = 0;
    
    u64 procLocal = 0;
    u64 procRemote = 0;
    u64 procIncoming = 0;
    
    if(depc) {
        neighbors = depv[0].ptr;
        neighborCount = paramv[2];
        u64 visitMask = (u64) -1; 
        localCount = processVertex(start, neighbors, neighborCount, &visitMask, &procIncoming, &procRemote);
        hiveAtomicAddU64(&otherCount, localCount);
        hiveAtomicAddU64(&incoming, procIncoming);
    }
    else {
        for(vertex i=start; i<end; i++) {
            u64 visitMask = 0;
            getNeighbors(&graph, i, &neighbors, &neighborCount);
            localCount += processVertex(i, neighbors, neighborCount, &visitMask, &procLocal, &procRemote);
        }
        hiveAtomicAddU64(&localTriangleCount, localCount);
        hiveAtomicAddU64(&local, procLocal);
        hiveAtomicAddU64(&remote, procRemote);
    }
    
    
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
    initBlockDistributionWithCmdLineArgs(&distribution, argc, argv);
    loadGraphUsingCmdLineArgs(&graph, &distribution, argc, argv);

    startReduceGuid = hiveReserveGuidRoute(HIVE_EDT,   0);
    finalReduceGuid = hiveReserveGuidRoute(HIVE_EDT,   0);
    epochGuid       = hiveInitializeEpoch(0, startReduceGuid, 0);

    distStart = nodeStart(nodeId, &distribution);
    distEnd   = nodeEnd(nodeId, &distribution);
    blockSize = (nodeEnd(nodeId, &distribution) - nodeStart(nodeId, &distribution)) / (hiveGetTotalWorkers() * overSub);
    if(!blockSize)
        blockSize = 1;
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {
    if(!nodeId && !workerId) {
        time = hiveGetTimeStamp();
        hiveEdtCreateWithGuid(startReduce, startReduceGuid, 0, NULL, 1);
        hiveEdtCreateWithGuid(finalReduce, finalReduceGuid, 0, NULL, hiveGetTotalNodes());
    }
    
    hiveStartEpoch(epochGuid);
    
    u64 args[2];
    u64 workerIndex = 0;
    for (vertex i = distStart; i < distEnd; i+=blockSize) {
        if(workerIndex % hiveGetTotalWorkers() == workerId) {
            args[0] = i;
            args[1] = (i+blockSize < distEnd) ? i+blockSize : distEnd;
            hiveEdtCreate(visitVertex, nodeId, 2, args, 0);
        }
        workerIndex++;
    }
}

int main(int argc, char** argv) {
    hiveRT(argc, argv);
    return 0;
}
