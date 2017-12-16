#ifndef HIVE_BLOCK_DISTRIBUTION_H
#define HIVE_BLOCK_DISTRIBUTION_H

#include "graph_defs.h"

struct hiveBlockDistribution {
  graph_sz_t num_vertices; // the complete number of vertices
  graph_sz_t num_edges; // the complete number of edges
  graph_sz_t block_sz;
  hiveGuid_t * graphGuid;
};

typedef struct hiveBlockDistribution hive_block_dist_t;

graph_sz_t getBlockSize(const hive_block_dist_t* const _dist);
graph_sz_t getNodeBlockSize(node_t _node, const hive_block_dist_t* const _dist);
void initBlockDistribution(hive_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m);
node_t getOwner(vertex v, const hive_block_dist_t* const _dist);
vertex nodeStart(node_t n, const hive_block_dist_t* const _dist);
vertex nodeEnd(node_t n, const hive_block_dist_t* const _dist);

local_index_t getLocalIndex(vertex v, const hive_block_dist_t* const _dist);

// Note : This is always for current rank
// TODO should remove local arg. Added this because linker was complaining
// about hiveGlobalRankId. Too tired to debug the problem ....
vertex getVertexId(node_t local, local_index_t, const hive_block_dist_t* const _dist);

void initBlockDistributionWithCmdLineArgs(hive_block_dist_t* _dist,
                                          int argc, 
                                          char** argv);

void freeDistribution(hive_block_dist_t* _dist);

hiveGuid_t* getGuidForVertex(vertex v,
                             const hive_block_dist_t* const _dist);

hiveGuid_t* getGuidForCurrentNode(const 
                                  hive_block_dist_t* const _dist);

#endif


