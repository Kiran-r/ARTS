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
#include <inttypes.h>
#include <assert.h>
#include "arts.h"
#include "artsGraph.h"

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
  arts_block_dist_t dist;
  initBlockDistribution(&dist, 
                        8, /*global vertices*/ 
                        11); /*global edges*/


  // Create a list of edges, use artsEdgeVector
  artsEdgeVector vec;
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

  vertex* neighbors = NULL;
  graph_sz_t nbrcnt = 0;
  getNeighbors(&graph, (vertex)1, &neighbors, &nbrcnt);
  assert(nbrcnt == 6);

  PRINTF("Neighbors of 1 : {"); 
  for (graph_sz_t i =0; i < nbrcnt; ++i) {
    PRINTF("%" PRIu64 ", ", neighbors[i]);
  }
  PRINTF("}"); 
  

  freeCSR(&graph);

  // Testing -- reading from commandline
  // e.g., mpirun -np 1 ./testCSR --file /Users/kane972/Downloads/ca-HepTh.tsv --num-vertices 9877 --num-edges 51946 --keep-self-loops

  arts_block_dist_t distCmd;
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
  artsRT(argc, argv);
  return 0;
}
