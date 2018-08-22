#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "artsRT.h"

#include <signal.h>
#include "artsGraph.h"
#include "artsTerminationDetection.h"
#include "shadAdapter.h"

#include<iostream>
#include<set>
#include <algorithm>
#include<vector>

arts_block_dist_t distribution;
csr_graph graph;
char* _file = NULL;
artsGuid_t maxReducerGuid = NULL_GUID;

u64 startTime;
u64 endTime;

typedef struct {
  artsGuid_t findIntersectionGuid;
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

// artsGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]);

// artsGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
//   printf("Called exit\n");
//   artsShutdown();
// }

artsGuid_t maxReducer(u32 paramc, u64 * paramv,
				     u32 depc, artsEdtDep_t depv[]) {
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
  endTime = artsGetTimeStamp();
  printf("Total execution time: %f s \n", (double)(endTime - startTime)/1000000000.0);
  artsStopIntroShad();
  artsShutdown();
}

artsGuid_t findIntersection(u32 paramc, u64 * paramv,
				     u32 depc, artsEdtDep_t depv[]) {
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
  artsGuid_t dbGuid = artsDbCreate(&ptr, dbSize, ARTS_DB_READ);
  perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
  vertexScanStat->source = source;
  vertexScanStat->scanStat = sum;
  // std::cout << "Source " << source << " ScanStat: " << sum << std::endl;
  artsSignalEdt(maxReducerGuid, source, dbGuid);
}

artsGuid_t visitOneHopNeighborOnRank(u32 paramc, u64 * paramv,
				     u32 depc, artsEdtDep_t depv[]) {
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;
  vertex* oneHopNeighbor;
  vertex* immediateNeighbors = srcInfo->neighbors;
  std::vector<vertex> localIntersection;
  for (unsigned int i = 0; i < srcInfo->numNeighbors; i++) {
    vertex current_neighbor = (vertex) srcInfo->neighbors[i];
    if (getOwner(current_neighbor, &distribution) == artsGetCurrentNode()) {
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
  artsGuid_t dbGuid = artsDbCreate(&ptr, dbSize, ARTS_DB_READ);
  perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
  vertexScanStat->source = srcInfo->source;
  vertexScanStat->scanStat = localIntersection.size();
  // std::cout << "Source: " << srcInfo->source << " rank: "  << artsGetCurrentNode() << " set intersection size: " << localIntersection.size() <<std::endl;
  artsSignalEdt(srcInfo->findIntersectionGuid, artsGetCurrentNode(), dbGuid);
}

artsGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) { 
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  vertex source = (vertex) paramv[0];
  // std::cout << "Visiting source: " << source <<std::endl;
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    /*Now spawn an edt that will wait to get oneHopneighbors from all the ranks in slots and calculate the grand count */
    artsGuid_t findIntersectionGuid = artsEdtCreate(findIntersection, artsGetCurrentNode(), 0, NULL, artsGetTotalNodes());
     /*For each rank, now spawn an edt that will perform an intersection*/
    for (unsigned int i = 0; i < artsGetTotalNodes(); i++) {
      unsigned int dbSize = sizeof (sourceInfo) + (sizeof(vertex) * neighbor_cnt);
      void * ptr = NULL;
      artsGuid_t dbGuid = artsDbCreate(&ptr, dbSize, ARTS_DB_READ);
      sourceInfo * srcInfo = (sourceInfo *) ptr;
      srcInfo->findIntersectionGuid = findIntersectionGuid;
      srcInfo->source = source;
      srcInfo->numNeighbors = neighbor_cnt;
      memcpy(&(srcInfo->neighbors), neighbors, sizeof(vertex) * neighbor_cnt);
      /*create the edt to find # one-hop neighbors*/
      artsGuid_t visitOneHopNeighborGuid = artsEdtCreate(visitOneHopNeighborOnRank, i, 0, NULL, 1);
      artsSignalEdt(visitOneHopNeighborGuid, 0, dbGuid);
   }
  } else {
    /*signal maxreducer*/
    unsigned int dbSize =  sizeof(perVertexScanStat);
    void * ptr = NULL;
    artsGuid_t dbGuid = artsDbCreate(&ptr, dbSize, ARTS_DB_READ);
    perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
    vertexScanStat->source = source;
    vertexScanStat->scanStat = 1;
    // std::cout << "signaling maxruducer for source " << source << std::endl;
    artsSignalEdt(maxReducerGuid, source, dbGuid);
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
  maxReducerGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

/*TODO: How to start parallel vertex scan stat calculation? How to do an efficient max reduction?*/
extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId,
		   int argc, char** argv) {
  printf("Node %u argc %u\n", nodeId, argc);
  if (!nodeId && !workerId) {
    /*This edt will calculate which vertex has the maximally induced subgraph.*/
    artsEdtCreateWithGuid(maxReducer, maxReducerGuid, 0, NULL, distribution.num_vertices);
    // artsGuid_t exitGuid = artsEdtCreate(exitProgram, 0, 0, NULL, 1);    
    // artsInitializeAndStartEpoch(exitGuid, 0);
    artsStartIntroShad(5);
    startTime = artsGetTimeStamp();
    for (uint64_t i = 0; i < distribution.num_vertices; ++i) {
      uint64_t source = i;  
      node_t rank = getOwner(source, &distribution);
      u64 packed_values[1] = {source};
      artsGuid_t visitSourceGuid = artsEdtCreate(visitSource, rank, 1, (u64*) &packed_values, 1);
      artsSignalEdtValue(visitSourceGuid, -1, 0);
    }
    // artsShutdown();
  }
}
int main(int argc, char** argv) {
  // raise(SIGTRAP);
    artsRT(argc, argv);
    return 0;
}
