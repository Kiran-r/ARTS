#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"

#include<set>
#include<vector>
#include <algorithm>
#include <iterator>

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

int compare(const void * a, const void * b)
{
  return ( *(u64*)a - *(u64*)b );
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    hiveShutdown();
}

hiveGuid_t maxReducer(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
}

hiveGuid_t aggregateNeighbors(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  std::set<vertex> grandUnion;
  
  /*first create the grand union with one hop neighbors*/
  for (u32 rank = 0; rank < depc; rank++) {
    sourceInfo * localUnion = (sourceInfo *) depv[rank].ptr;
    for (u32 i = 0; i < localUnion->numNeighbors; i++) {
      grandUnion.insert(localUnion->neighbors[i]);
    }
  }
  /*Then add immediate neighbors to the grand union*/
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;    
  vertex source = (vertex) srcInfo->source;
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  for (u32 i = 0; i < neighbor_cnt; i++) {
    grandUnion.insert(neighbors[i]);
  }


}

hiveGuid_t visitOneHopNeighborOnRank(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  sourceInfo * srcInfo = (sourceInfo *) depv[0].ptr;
  vertex* oneHopNeighbor;
  std::set<vertex> localUnion;
  for (unsigned int i = 0; i <srcInfo->numNeighbors; i++) {
    vertex current_neighbor = (vertex) srcInfo->neighbors[i];
    vertex* oneHopNeighbors = NULL;
    u64 neighbor_cnt = 0;
    getNeighbors(&graph, current_neighbor, &oneHopNeighbors, &neighbor_cnt);
    // std::vector<vertex> oneHopNeighborsV;
    for (unsigned int j = 0; j < neighbor_cnt; j++) {
      localUnion.insert(oneHopNeighbors[j]);
      //oneHopNeighborsV.push_back(oneHopNeighbors[j]);
    }
    // std::sort(oneHopNeighborsV.begin(), oneHopNeighborsV.end());
    
    // qsort(&oneHopNeighbors, neighbor_cnt, sizeof(vertex), compare);
  }

  /*Not sure about data serialization.*/
  // create  a db containing the local union
  unsigned int dbSize = localUnion.size() * sizeof(vertex);
  void * ptr = NULL;
  hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
  sourceInfo * srcAndLocalUnionInfo = (sourceInfo *) ptr;
  srcAndLocalUnionInfo->source = srcInfo->source;
  srcAndLocalUnionInfo->numNeighbors = localUnion.size();
  memcpy(srcAndLocalUnionInfo->neighbors, &localUnion, dbSize);
  hiveSignalEdt(srcInfo->aggregateNeighborGuid, dbGuid, 
		hiveGetCurrentNode(), DB_MODE_ONCE);
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) { 
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  vertex source = (vertex) paramv[0];
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    qsort(&neighbors, neighbor_cnt, sizeof(graph_sz_t), compare);
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
      hiveSignalEdt(visitOneHopNeighborGuid, dbGuid, 0, DB_MODE_ONCE);
   }

   
    // TODO: pass around the visitOneHopNeighborGuid also vvv?
    /* hiveSignalEdt(visitOneHopNeighborGuid, dbGuid, neighbor_cnt, DB_MODE_ONCE_LOCAL); */
    /* for (unsigned int i = 0; i < neighbor_cnt; i++) { */
    /*   vertex neib = neighbors[i]; */
    /*   node_t rank = getOwner(neib, &distribution); */
    /*   hiveGuid_t getOneHopNeighborCountGuid = hiveEdtCreate(getOneHopNeighborCount, rank, 0, NULL, 1); */
    /*   void * positionPtr = NULL; */
    /*   hiveGuid_t positionGuid = hiveDbCreate(&positionPtr, dbSize, false); */
    /*   u64 *pos = positionPtr; */
    /*   pos = &i;  */
    /*   hiveSignalEdt(getOneHopNeighborCountGuid, positionPtr, 0, DB_MODE_ONCE_LOCAL;) */
    /* } */
  }
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {

  // This is the dbGuid we will need to aquire to do gets and puts to the score property arrayDb
  // vertexPropertyMapGuid = hiveReserveGuidRoute(HIVE_DB, 0);
  // vertexIDMapGuid = hiveReserveGuidRoute(HIVE_DB, 0);

  // distribution must be initialized in initPerNode
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
void initPerWorker(unsigned int nodeId, unsigned int workerId,
		   int argc, char** argv) {
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
  }
}
int main(int argc, char** argv) {
    hiveRT(argc, argv);
    return 0;
}
