#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"

#define DPRINTF(...)
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

hive_block_dist_t distribution;
csr_graph graph;
u64* level;

void bfs_output() {
  DPRINTF("Printing vertex levels....\n");
  u64 i;
  for(i=0; i < graph.num_local_vertices; ++i) {
    PRINTF("Local vertex : %" PRIu64 ", Level : %" PRIu64 "\n", i, level[i]);
  }
}

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  bfs_output();
  hiveShutdown();
}

void bfs_send(vertex u, u64 ulevel);

hiveGuid_t relax(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  DPRINTF("calling relax\n");
  assert(paramc == 2);
  vertex v = (vertex)paramv[0];
  u64 vlevel = paramv[1];
  
  local_index_t indexv = getLocalIndex(v, &distribution);
  assert(indexv < graph.num_local_vertices);

  u64 oldlevel = level[indexv];
  bool success = false;
  while(vlevel < oldlevel) {
    // NOTE : This call depends on GNU (GCC)
    success = __atomic_compare_exchange(&level[indexv],
                                        &oldlevel,
                                        &vlevel,
                                        false,
                                        __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED);
    oldlevel = level[indexv];
  }

  if (success) {
    // notify neighbors
    // get neighbors
    vertex* neighbors = NULL;
    u64 neighbor_cnt = 0;
    getNeighbors(&graph, v,
                 &neighbors,
                 &neighbor_cnt);

    // iterate over neighbors
    u64 neigbrlevel = level[indexv]+1;
    for(u64 i=0; i < neighbor_cnt; ++i) {        
      vertex u = neighbors[i];
      
      // route message
      DPRINTF("sending u=%" PRIu64 ", level= %" PRIu64 "\n", u, neigbrlevel);
      incrementActiveCount(1);
      bfs_send(u, neigbrlevel);
    }
  }
  incrementFinishedCount(1);
}

void bfs_send(vertex u, u64 ulevel) {
  hiveGuid_t* neighbDbguid = getGuidForVertex(u, &distribution);
  u64 send[2];
  send[0] = u;
  send[1] = ulevel;
  hiveGuid_t relaxGuid = hiveActiveMessageWithDb(relax,            // function 
                                                 2,                // number of parameters
                                                 send,             // parameters
                                                 0,                // additional deps
                                                 (*neighbDbguid)); // this is the guid to co-locate task with
}

hiveGuid_t kickoffTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  PRINTF("Kick off\n");
  incrementActiveCount(1);
  hiveGuid_t exitProgramGuid = hiveEdtCreateWithGuid(exitProgram, 0, 0, NULL, 1);
  hiveDetectTermination(exitProgramGuid, 0);
  
  vertex source = (vertex)paramv[0];
  bfs_send(source, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
    
  // distribution must be initialized in initPerNode
  initBlockDistributionWithCmdLineArgs(&distribution, 
                                       argc, argv);
  // set-up the graph
  loadGraphUsingCmdLineArgs(&graph,
			    &distribution,
			    argc,
			    argv);

  // should probably encapsulate into something
  level = (u64 *)hiveMalloc(graph.num_local_vertices * sizeof(u64));
  // initialize the level array
  for (u64 i=0; i < graph.num_local_vertices; ++i) {
    level[i] = UINT64_MAX;
  }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) {   
  
  if (!workerId) {
    // find the source vertex
    vertex source;
    for (int i=0; i < argc; ++i) {
      if (strcmp("--source", argv[i]) == 0) {
        sscanf(argv[i+1], "%" SCNu64, &source);
      }
    }

    assert(source < distribution.num_vertices);

    if(!nodeId)
    {
        hiveGuid_t startGuid = hiveEdtCreate(kickoffTermination, 0, 1, (u64*) &source, hiveGetTotalNodes());
        initializeTerminationDetection(startGuid);
    }
  }
}

int main(int argc, char** argv)
{
  hiveRT(argc, argv);
  return 0;
}
