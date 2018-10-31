//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "arts.h"
#include "artsGraph.h"
#include "artsAtomics.h"

arts_block_dist_t distribution;
csr_graph graph;

artsGuid_t epochGuid       = NULL_GUID;
artsGuid_t startReduceGuid = NULL_GUID;
artsGuid_t finalReduceGuid = NULL_GUID;

vertex distStart = 0;
vertex distEnd   = 0;
uint64_t blockSize    = 0;
uint64_t overSub      = 16;

uint64_t otherCount = 0;
uint64_t localTriangleCount = 0;
uint64_t time = 0;

uint64_t local = 0;
uint64_t remote = 0;
uint64_t incoming = 0;

void finalReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void localReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void startReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
void visitVertex(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);

//Only support up to 64 nodes
inline unsigned int checkAndSet(uint64_t * mask, unsigned int index) {
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
    PRINTF("Local: %lu Remote: %lu Incoming: %lu\n", local, remote, incoming);
    artsSignalEdtValue(finalReduceGuid, artsGetCurrentNode(), localTriangleCount+otherCount);
}

void startReduce(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
        artsEdtCreateDep(localReduce, i, 0, NULL, 0, false);
    }
}

inline uint64_t lowerBound(vertex value, uint64_t start, uint64_t end, vertex * edges) {
    while ((start < end) && (edges[start] < value))
        start++;
    return start;
}

inline uint64_t upperBound(vertex value, uint64_t start, uint64_t end, vertex * edges) {
    while ((start < end) && (value < edges[end - 1]))
        end--;
    return end;
}

inline uint64_t countTriangles(vertex * a, uint64_t a_start, uint64_t a_end, vertex * b, uint64_t b_start, uint64_t b_end) {
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

inline uint64_t processVertex(vertex i, vertex * neighbors, uint64_t neighborCount, uint64_t * visitMask, uint64_t * procLocal, uint64_t * procRemote) {
    uint64_t localCount = 0;
    
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
            localCount += countTriangles(neighbors, firstPred, nextPred, jNeighbors, firstSucc, lastSucc);
            (*procLocal)++;
        }
        else if(checkAndSet(visitMask, owner)) {
            uint64_t args[3];
            args[0] = i;
            args[1] = i;
            args[2] = neighborCount;
            artsGuid_t guid = artsEdtCreate(visitVertex, owner, 3, args, 1);
            artsSignalEdtPtr(guid, 0, neighbors, sizeof(vertex) * neighborCount);
            (*procRemote)++;
        }
    }
    return localCount;
}

void visitVertex(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) {
    uint64_t localCount = 0;
    
    vertex start = paramv[0];
    vertex end   = paramv[1];
    
    vertex * neighbors = NULL;
    uint64_t neighborCount = 0;
    
    uint64_t procLocal = 0;
    uint64_t procRemote = 0;
    uint64_t procIncoming = 0;
    
    if(depc) {
        neighbors = depv[0].ptr;
        neighborCount = paramv[2];
        uint64_t visitMask = (uint64_t) -1; 
        localCount = processVertex(start, neighbors, neighborCount, &visitMask, &procIncoming, &procRemote);
        artsAtomicAddU64(&otherCount, localCount);
        artsAtomicAddU64(&incoming, procIncoming);
    }
    else {
        for(vertex i=start; i<end; i++) {
            uint64_t visitMask = 0;
            getNeighbors(&graph, i, &neighbors, &neighborCount);
            localCount += processVertex(i, neighbors, neighborCount, &visitMask, &procLocal, &procRemote);
        }
        artsAtomicAddU64(&localTriangleCount, localCount);
        artsAtomicAddU64(&local, procLocal);
        artsAtomicAddU64(&remote, procRemote);
    }
    
    
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
    initBlockDistributionWithCmdLineArgs(&distribution, argc, argv);
    loadGraphUsingCmdLineArgs(&graph, &distribution, argc, argv);

    startReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    finalReduceGuid = artsReserveGuidRoute(ARTS_EDT,   0);
    epochGuid       = artsInitializeEpoch(0, startReduceGuid, 0);

    distStart = nodeStart(nodeId, &distribution);
    distEnd   = nodeEnd(nodeId, &distribution);
    blockSize = (nodeEnd(nodeId, &distribution) - nodeStart(nodeId, &distribution)) / (artsGetTotalWorkers() * overSub);
    if(!blockSize)
        blockSize = 1;
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {
    if(!nodeId && !workerId) {
        time = artsGetTimeStamp();
        artsEdtCreateWithGuid(startReduce, startReduceGuid, 0, NULL, 1);
        artsEdtCreateWithGuid(finalReduce, finalReduceGuid, 0, NULL, artsGetTotalNodes());
    }
    
    artsStartEpoch(epochGuid);
    
    uint64_t args[2];
    uint64_t workerIndex = 0;
    for (vertex i = distStart; i < distEnd; i+=blockSize) {
        if(workerIndex % artsGetTotalWorkers() == workerId) {
            args[0] = i;
            args[1] = (i+blockSize < distEnd) ? i+blockSize : distEnd;
            artsEdtCreate(visitVertex, nodeId, 2, args, 0);
        }
        workerIndex++;
    }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}
