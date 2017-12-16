#ifndef HAGGLE_CSR_H
#define HAGGLE_CSR_H

#include "graph_defs.h"
#include "block_distribution.h"
#include "hiveEdgeVector.h"

#define MAXCHAR 1000

typedef struct {
  graph_sz_t num_local_vertices;
  graph_sz_t num_local_edges;
  vertex* row_indices;
  vertex* columns;
  hive_block_dist_t* distribution;
  void* data;
} csr_graph;


void initCSR(csr_graph* _csr, 
             graph_sz_t _localv,
             graph_sz_t _locale,
             hive_block_dist_t* _dist,
             hiveEdgeVector* _edges,
             bool _sorted_by_src);

int loadGraphNoWeight(const char* _file,
                      csr_graph* _graph,
                      hive_block_dist_t* _dist,
                      bool _flip,
                      bool _ignore_self_loops);

void printLocalCSR(const csr_graph* _csr);

int loadGraphUsingCmdLineArgs(csr_graph* _graph,
                              hive_block_dist_t* _dist,
                              int argc, char** argv);
void freeCSR(csr_graph* _csr);

void getNeighbors(csr_graph* _csr,
                  vertex v,
                  vertex** _out,
                  graph_sz_t* _neighborcount);
#endif
