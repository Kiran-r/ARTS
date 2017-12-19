#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "hiveGraph.h"

void initPerNode(unsigned int nodeId, int argc, char** argv) {
  // Simple Graph, vertices = 8, edges = 11
  /**
     0 6
     1 2
     2 5
     2 3
     2 4
     1 6
     1 3
     1 7
     1 4
     3 5
     1 5
  **/

  // Question : Should the following code go inside init node or
  // should it go in initWorker with "if (!nodeid && !workerid) {...} ?
  // Clarify with Josh.

  int edge_arr[] = {
    5,  6,
    1,  2,
    2,  5,
    2,  3,
    2,  4,
    1,  6,
    1,  3,
    1,  7,
    1,  4,
    3,  5,
    1, 5
  };


  // Create a block distribution
  hive_block_dist_t dist;
  initBlockDistribution(&dist, 
                        8, /*global vertices*/ 
                        11); /*global edges*/


  // Create a list of edges, use hiveEdgeVector
  hiveEdgeVector vec;
  initEdgeVector(&vec, 100);
  for(int i=0; i < 11; ++i) {
    pushBackEdge(&vec, edge_arr[i*2], edge_arr[(i*2)+1], 0);
  }

  // Create the CSR graph, graphGuid is used to allocate
  // row indices and column array
  csr_graph graph;
  initCSR(&graph, // graph structure
          8, // number of "local" vertices
          11, // number of "local" edges
          &dist, // distribution
          &vec, // edges
          false /*are edges sorted ?*/);

  // Edge list not needed after creating the CSR
  freeEdgeVector(&vec);

  printLocalCSR(&graph);

  freeCSR(&graph);

  // Testing -- reading from commandline
  // e.g., mpirun -np 1 ./testCSR --file /Users/kane972/Downloads/ca-HepTh.tsv --num-vertices 9877 --num-edges 51946 --keep-self-loops

  hive_block_dist_t distCmd;
  initBlockDistributionWithCmdLineArgs(&distCmd, 
                                       argc, argv);

  csr_graph graphCmd;
  loadGraphUsingCmdLineArgs(&graphCmd,
                            &distCmd,
                            argc,
                            argv);

  printLocalCSR(&graphCmd);

  freeCSR(&graphCmd);
  //  const char* fname = "/Users/kane972/Downloads/wiki-Vote.txt";
  //loadGraphNoWeight(fname, &graphGuid, &graph, &dist);

  //  PRINTF("Initilization per node\n");
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   

}

int main(int argc, char** argv)
{

  hiveRT(argc, argv);
  return 0;
}