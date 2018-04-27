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

#include<iostream>
#include<set>
#include <algorithm>
#include<vector>

hive_block_dist_t distribution;
csr_graph graph;
char* _file = NULL;
hiveGuid_t maxReducerGuid = NULL_GUID;

typedef struct {
  hiveGuid_t aggregateNeighborGuid;
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

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  printf("Called exit\n");
  hiveShutdown();
}

hiveGuid_t maxReducer(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  u32 maxScanStat = 0;
  vertex maxVertex = 0;
  for (u32 v = 0; v < depc; v++) {
    perVertexScanStat * vertexScanStat = (perVertexScanStat *) depv[v].ptr;
    std::cout << "Vertex: " << vertexScanStat->source << " scan_stat: " << vertexScanStat->scanStat << std::endl;
    if (vertexScanStat->scanStat > maxScanStat) {
      maxScanStat = vertexScanStat->scanStat;
      maxVertex = vertexScanStat->source;
    }
  }
  std::cout << "Max vertex: " << maxVertex << "scanStat: " << maxScanStat << std::endl;

}

hiveGuid_t aggregateNeighbors(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  // std::set<vertex> grandUnion;
  std::vector<vertex> grandUnion;

  /*first create the grand union with one hop neighbors*/
  for (u32 rank = 0; rank < depc; rank++) {
    sourceInfo * localUnion = (sourceInfo *) depv[rank].ptr;
    // for (u32 i = 0; i < localUnion->numNeighbors; i++) {
    //   grandUnion.insert(localUnion->neighbors[i]);
    // }
    std::set_union(grandUnion.begin(), grandUnion.end(),
	       &(localUnion->neighbors[0]), 
	       &(localUnion->neighbors[localUnion->numNeighbors]),
	       std::back_inserter(grandUnion));
  }

  /*Then add immediate neighbors to the grand union*/
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;    
  vertex source = (vertex) srcInfo->source;
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  // for (u32 i = 0; i < neighbor_cnt; i++) {
  //   grandUnion.insert(neighbors[i]);
  // }
  std::set_union(grandUnion.begin(), grandUnion.end(),
		 neighbors, neighbors + neighbor_cnt,
		 std::back_inserter(grandUnion));
  /*insert itself*/
  grandUnion.push_back(source);

  /*Now retain only unique elements*/
  std::sort(grandUnion.begin(), grandUnion.end());
  auto last = std::unique(grandUnion.begin(), grandUnion.end());
  grandUnion.resize(std::distance(grandUnion.begin(), last));
  

  unsigned int dbSize =  sizeof(perVertexScanStat);
  void * ptr = NULL;
  hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
  perVertexScanStat * vertexScanStat = (perVertexScanStat *) ptr;  
  vertexScanStat->source = source;
  vertexScanStat->scanStat = grandUnion.size();
  std::cout << "Aggregate size for source: " << source << " is " << grandUnion.size() << std::endl;
  for (std::vector<vertex>::iterator i = grandUnion.begin(); i < grandUnion.end(); i++) {
    std::cout  << *i << " " ;
  }
  std::cout << std::endl;
  hiveSignalEdt(maxReducerGuid, dbGuid, source, DB_MODE_ONCE_LOCAL);
}

hiveGuid_t visitOneHopNeighborOnRank(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;
  vertex* oneHopNeighbor;
  // std::set<vertex> localUnion;
  std::vector<vertex> localUnion;
  for (unsigned int i = 0; i <srcInfo->numNeighbors; i++) {
    vertex current_neighbor = (vertex) srcInfo->neighbors[i];
    std::cout << "current_neighbor: " << current_neighbor << std::endl;
    vertex* oneHopNeighbors = NULL;
    u64 neighbor_cnt = 0;
    getNeighbors(&graph, current_neighbor, &oneHopNeighbors, &neighbor_cnt);
    for (unsigned int j = 0; j < neighbor_cnt; j++) {
      std::cout << "One-hop neighbor for " <<  srcInfo->source << " is: " << oneHopNeighbors[j] << std::endl;
    }
    std::set_union(localUnion.begin(), localUnion.end(), 
		   oneHopNeighbors, oneHopNeighbors + neighbor_cnt,
		   std::back_inserter(localUnion));
    // for (unsigned int j = 0; j < neighbor_cnt; j++) {
    //   localUnion.insert(oneHopNeighbors[j]);
    // }
  }

  /*Now retain only unique elements*/
  std::sort(localUnion.begin(), localUnion.end());
  auto last = std::unique(localUnion.begin(), localUnion.end());
  localUnion.resize(std::distance(localUnion.begin(), last));

  // /*Not sure about data serialization.*/
  // create  a db containing the local union
  unsigned int dbSize = sizeof(sourceInfo) + localUnion.size() * sizeof(vertex);
  void * ptr = NULL;
  hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
  sourceInfo * srcAndLocalUnionInfo = (sourceInfo *) ptr;
  srcAndLocalUnionInfo->source = srcInfo->source;
  srcAndLocalUnionInfo->numNeighbors = localUnion.size();
  std::copy(localUnion.begin(), localUnion.end(), srcAndLocalUnionInfo->neighbors);
  hiveSignalEdt(srcInfo->aggregateNeighborGuid, dbGuid, 
                hiveGetCurrentNode(), DB_MODE_ONCE_LOCAL);
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) { 
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  vertex source = (vertex) paramv[0];
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    // qsort(&neighbors, neighbor_cnt, sizeof(graph_sz_t), compare);
    int * numNeighborPerRank = (int *) calloc (hiveGetTotalNodes(), sizeof(int));

    /*Count how many neighbours per rank*/
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      unsigned int neighborRank = getOwner(neighbors[i], &distribution);
      numNeighborPerRank[neighborRank] += 1;
    }

    /*Based on the count, allocate memory for each rank's neighbors*/
    vertex **neighborsPerRank = (vertex **) malloc (hiveGetTotalNodes() * sizeof (vertex*));
    for (unsigned int i = 0; i < hiveGetTotalNodes(); i++) {
      neighborsPerRank[i] = (vertex *) malloc (numNeighborPerRank[i] * sizeof (vertex));
    }
    int * currentNumNeighborPerRank = (int *) calloc (hiveGetTotalNodes(), sizeof(int));

    /*Gather neighbors per rank*/
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      std:: cout << "Source: " << source << " Neighbor: " << neighbors[i] << std::endl;
      unsigned int neighborRank = getOwner(neighbors[i], &distribution);
      int currentNumOfNeighbor =  currentNumNeighborPerRank[neighborRank];
      neighborsPerRank[neighborRank][currentNumOfNeighbor] = neighbors[i];
      currentNumNeighborPerRank[neighborRank]++;
    }

    /*Now spawn an edt that will wait to get oneHopneighbors from all the ranks in slots and calculate the grand union */
    hiveGuid_t aggregateNeighborsGuid = hiveEdtCreate(aggregateNeighbors, hiveGetCurrentNode(), 0, NULL, hiveGetTotalNodes());
 
    /*For each rank, now spawn an edt that will perform local union with all the neighbors first and then send back the local union*/
    for (unsigned int i = 0; i < hiveGetTotalNodes(); i++) {
      unsigned int dbSize = sizeof (sourceInfo) + sizeof(vertex) * numNeighborPerRank[i];
      void * ptr = NULL;
      hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
      sourceInfo * srcInfo = (sourceInfo *) ptr;
      srcInfo->aggregateNeighborGuid = aggregateNeighborsGuid;
      srcInfo->source = source;
      srcInfo->numNeighbors = numNeighborPerRank[i];
      memcpy(srcInfo->neighbors, neighborsPerRank[i], sizeof(vertex) * numNeighborPerRank[i]);
      /*create the edt to find # one-hop neighbors*/
      /*Can we use the same guid again and agin?*/
      hiveGuid_t visitOneHopNeighborGuid = hiveEdtCreate(visitOneHopNeighborOnRank, i, 0, NULL, 1);
      hiveSignalEdt(visitOneHopNeighborGuid, dbGuid, 0, DB_MODE_ONCE_LOCAL);
   }
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
  maxReducerGuid = hiveEdtCreate(maxReducer, 0, 0, NULL, distribution.num_vertices);
}

/*TODO: How to start parallel vertex scan stat calculation? How to do an efficient max reduction?*/
extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId,
		   int argc, char** argv) {
  printf("Node %u argc %u\n", nodeId, argc);
  if (!nodeId && !workerId) {
    hiveGuid_t exitGuid = hiveEdtCreate(exitProgram, 0, 0, NULL, 1);    
    hiveInitializeAndStartEpoch(exitGuid, 0);
    for (uint64_t i = 0; i < distribution.num_vertices; ++i) {
      uint64_t source = i;  
      node_t rank = getOwner(source, &distribution);
      u64 packed_values[1] = {source};
      hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 1, (u64*) &packed_values, 1);
      hiveSignalEdt(visitSourceGuid, 0, 0, DB_MODE_SINGLE_VALUE);
    }
    // hiveShutdown();
  }
}
int main(int argc, char** argv) {
  // raise(SIGTRAP);
    hiveRT(argc, argv);
    return 0;
}
