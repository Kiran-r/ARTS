#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"

hive_block_dist_t distribution;
csr_graph graph;

u64* level;
hiveGuid_t relaxGuid = NULL_GUID;

hiveGuid_t kickoffTerminationGuid = NULL_GUID;
hiveGuid_t exitProgramGuid = NULL_GUID;

void initPerNode(unsigned int nodeId, int argc, char** argv) {

  // distribution must be initilized in initPerNode
  initBlockDistributionWithCmdLineArgs(&distribution, 
                                       argc, argv);
  
  relaxGuid = hiveReserveGuidRoute(HIVE_EDT, 0);

}

void bfs_output() {
  fprintf(stderr, "Printing vertex levels....\n");
  u64 i;
  for(i=0; i < graph.num_local_vertices; ++i) {
    printf("Local vertex : %" PRIu64 ", Level : %" PRIu64 "\n", i, level[i]);
  }
}

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]){
  bfs_output();
  hiveShutdown();
}

hiveGuid_t kickoffTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  exitProgramGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  hiveEdtCreateWithGuid(exitProgram, exitProgramGuid, 0, NULL, 1);
  hiveDetectTermination(exitProgramGuid, 0); 
}

// bit like mesage passing
void bfs_send(vertex, u64);

hiveGuid_t relax(u32 paramc, u64 * paramv, 
                 u32 depc, 
                 hiveEdtDep_t depv[]) {

  //fprintf(stderr, "calling relax\n");
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

      incrementActiveCount(1);
      // route message
      //fprintf(stderr, "2sending u=%" PRIu64 ", level= %" PRIu64 "\n", u, level[indexv]);
      bfs_send(u, neigbrlevel);
    }
  }

  incrementFinishedCount(1);
}

void bfs_send(vertex u,
          u64 ulevel) {

  hiveGuid_t* neighbDbguid = getGuidForVertex(u, &distribution);
  // Question : Understand what each of the parameters do
  // Need to send the neigbor vertex and the distance
  // Prbably there must be a better way to wrap parameters
  u64 send[2];
  send[0] = u;
  send[1] = ulevel;
  // Create the EDT
  hiveEdtCreateWithGuid(relax, //function 
                        relaxGuid,  // function guid
                        2, // number of parameters ?
                        send, // parameters
                        0);// no idea what this is!
  // Signal the EDT
  hiveSignalEdt(relaxGuid, 
                (*neighbDbguid), // relax function will get executed in this location
                0, // what is slot ?
                DB_MODE_NON_COHERENT_READ);// what is access mode ?

}


void initPerWorker(unsigned int nodeId, 
                   unsigned int workerId, 
                   int argc, char** argv) {   

  if (!nodeId && !workerId) {
    kickoffTerminationGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(kickoffTermination, kickoffTerminationGuid, 0, NULL, hiveGetTotalNodes());
    initializeTerminationDetection(kickoffTerminationGuid);
  }
  
  if (!workerId) {
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

    // find the source vertex
    vertex source;
    for (int i=0; i < argc; ++i) {
      if (strcmp("--source", argv[i]) == 0) {
        sscanf(argv[i+1], "%" SCNu64, &source);
      }
    }

    assert(source < distribution.num_vertices);
    // WE NEED A BARRIER HERE IN PARALLEL EXECUTION


    // is this source belong to current rank ?
    if (getOwner(source, &distribution) == hiveGetCurrentNode()) {
      // set level to zero
      // note: we need the local index
      incrementActiveCount(1);      
      fprintf(stderr, "sending %" PRIu64 "\n", source); 
      bfs_send(source, 0);     
    }

  }
  // HOW ARE WE GOING TO SHUTDOWN ? SEEMS LIKE WE NEED
  // SOME FORM TERMINATION FOR PARALLEL EXECUTION.
}

int main(int argc, char** argv)
{

  hiveRT(argc, argv);
  return 0;
}
