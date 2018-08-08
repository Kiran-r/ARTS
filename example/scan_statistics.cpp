#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"

#include <signal.h>
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"
#include "shadAdapter.h"

#include<iostream>
#include<set>
#include <algorithm>
#include<vector>

hive_block_dist_t distribution;
csr_graph graph;
char* _file = NULL;
hiveGuid_t maxReducerGuid = NULL_GUID;

u64 startTime;
u64 endTime;

typedef struct {
  hiveGuid_t findIntersectionGuid;
  vertex source;
  unsigned int numNeighbors;
  vertex neighbors[];
} sourceInfo;

typedef struct {
  vertex source;
  u64 scanStat;
} perVertexScanStat;

// int compare(const void * a, const void * b)
// {
//   return ( *(u64*)a - *(u64*)b );
// }

// hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

// hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
//   printf("Called exit\n");
//   hiveShutdown();
// }

hiveGuid_t maxReducer(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  // std::cout << "In max reducer" << std::endl;
  u32 maxScanStat = 0;
  vertex maxVertex = 0;
  for (u32 v = 0; v < depc; v++) {
    perVertexScanStat * vertexScanStat = (perVertexScanStat *) depv[v].ptr;
    // std::cout << "Vertex: " << vertexScanStat->source << " scan_stat: " << vertexScanStat->scanStat << std::endl;
    if (vertexScanStat->scanStat > maxScanStat) {
      maxScanStat = vertexScanStat->scanStat;
      maxVertex = vertexScanStat->source;
    }
  }
  std::cout << "Max vertex: " << maxVertex << " scanStat: " << maxScanStat << std::endl;
  endTime = hiveGetTimeStamp();
  printf("Total execution time: %f s \n", (double)(endTime - startTime)/1000000000.0);
  hiveStopIntroShad();
  hiveShutdown();
}

hiveGuid_t findIntersection(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  u64 sum = 0;
  perVertexScanStat * localIntersection = (perVertexScanStat *) depv[0].ptr;
  vertex source = (vertex) localIntersection->source;

  for (u64 rank = 0; rank < depc; rank++) {

    perVertexScanStat * localIntersection = (perVertexScanStat *) depv[rank].ptr;
    // std::cout << "Source: " << source << " Rank: " << rank << "Scanstat: " << localIntersection->scanStat << std::endl;
    sum +=  localIntersection->scanStat;
  }

  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  
  sum += neighbor_cnt;

  unsigned int dbSize =  sizeof(perVertexScanStat);
  void * ptr = NULL;
  hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
  perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
  vertexScanStat->source = source;
  vertexScanStat->scanStat = sum;
  // std::cout << "Source " << source << " ScanStat: " << sum << std::endl;
  hiveSignalEdt(maxReducerGuid, dbGuid, source, DB_MODE_NON_COHERENT_READ);
}

hiveGuid_t visitOneHopNeighborOnRank(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;
  vertex* oneHopNeighbor;
  vertex* immediateNeighbors = srcInfo->neighbors;
  std::vector<vertex> localIntersection;
  for (unsigned int i = 0; i < srcInfo->numNeighbors; i++) {
    vertex current_neighbor = (vertex) srcInfo->neighbors[i];
    if (getOwner(current_neighbor, &distribution) == hiveGetCurrentNode()) {
      // std::cout << "Source " << srcInfo->source << " Current_neighbor: " << current_neighbor << std::endl;
      vertex* oneHopNeighbors = NULL;
      u64 neighbor_cnt = 0;
      getNeighbors(&graph, current_neighbor, &oneHopNeighbors, &neighbor_cnt);
      for (unsigned int j = 0; j < neighbor_cnt; j++) {
	// std::cout << "One-hop neighbor for " <<  srcInfo->source << " is: " << oneHopNeighbors[j] << std::endl;
      }
      std::set_intersection(immediateNeighbors, 
			    immediateNeighbors + srcInfo->numNeighbors, 
			    oneHopNeighbors, oneHopNeighbors + neighbor_cnt,
			    std::back_inserter(localIntersection));
    }
  }

  unsigned int dbSize =  sizeof(perVertexScanStat);
  void * ptr = NULL;
  hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
  perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
  vertexScanStat->source = srcInfo->source;
  vertexScanStat->scanStat = localIntersection.size();
  // std::cout << "Source: " << srcInfo->source << " rank: "  << hiveGetCurrentNode() << " set intersection size: " << localIntersection.size() <<std::endl;
  hiveSignalEdt(srcInfo->findIntersectionGuid, dbGuid, hiveGetCurrentNode(), DB_MODE_NON_COHERENT_READ);
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) { 
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  vertex source = (vertex) paramv[0];
  // std::cout << "Visiting source: " << source <<std::endl;
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    /*Now spawn an edt that will wait to get oneHopneighbors from all the ranks in slots and calculate the grand count */
    hiveGuid_t findIntersectionGuid = hiveEdtCreate(findIntersection, hiveGetCurrentNode(), 0, NULL, hiveGetTotalNodes());
     /*For each rank, now spawn an edt that will perform an intersection*/
    for (unsigned int i = 0; i < hiveGetTotalNodes(); i++) {
      unsigned int dbSize = sizeof (sourceInfo) + (sizeof(vertex) * neighbor_cnt);
      void * ptr = NULL;
      hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
      sourceInfo * srcInfo = (sourceInfo *) ptr;
      srcInfo->findIntersectionGuid = findIntersectionGuid;
      srcInfo->source = source;
      srcInfo->numNeighbors = neighbor_cnt;
      memcpy(&(srcInfo->neighbors), neighbors, sizeof(vertex) * neighbor_cnt);
      /*create the edt to find # one-hop neighbors*/
      hiveGuid_t visitOneHopNeighborGuid = hiveEdtCreate(visitOneHopNeighborOnRank, i, 0, NULL, 1);
      hiveSignalEdt(visitOneHopNeighborGuid, dbGuid, 0, DB_MODE_NON_COHERENT_READ);
   }
  } else {
    /*signal maxreducer*/
    unsigned int dbSize =  sizeof(perVertexScanStat);
    void * ptr = NULL;
    hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
    perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
    vertexScanStat->source = source;
    vertexScanStat->scanStat = 1;
    // std::cout << "signaling maxruducer for source " << source << std::endl;
    hiveSignalEdt(maxReducerGuid, dbGuid, source, DB_MODE_NON_COHERENT_READ);
  }
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv) {
  // distribution must be initialized in initPerNode
  printf("Node %u argc %u\n", nodeId, argc);
  initBlockDistributionWithCmdLineArgs(&distribution,
  				       argc, argv);
  // read the edgelist and construct the graph
  loadGraphUsingCmdLineArgs(&graph,
  			    &distribution,
  			    argc,
  			    argv);
  maxReducerGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
}

/*TODO: How to start parallel vertex scan stat calculation? How to do an efficient max reduction?*/
extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId,
		   int argc, char** argv) {
  printf("Node %u argc %u\n", nodeId, argc);
  if (!nodeId && !workerId) {
    /*This edt will calculate which vertex has the maximally induced subgraph.*/
    hiveEdtCreateWithGuid(maxReducer, maxReducerGuid, 0, NULL, distribution.num_vertices);
    // hiveGuid_t exitGuid = hiveEdtCreate(exitProgram, 0, 0, NULL, 1);    
    // hiveInitializeAndStartEpoch(exitGuid, 0);
    hiveStartIntroShad(5);
    startTime = hiveGetTimeStamp();
    for (uint64_t i = 0; i < distribution.num_vertices; ++i) {
      uint64_t source = i;  
      node_t rank = getOwner(source, &distribution);
      u64 packed_values[1] = {source};
      hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 1, (u64*) &packed_values, 1);
      hiveSignalEdt(visitSourceGuid, 0, 0, DB_MODE_PIN);
    }
    // hiveShutdown();
  }
}
int main(int argc, char** argv) {
  // raise(SIGTRAP);
    hiveRT(argc, argv);
    return 0;
}
