#ifndef ARTS_BLOCK_DISTRIBUTION_H
#define ARTS_BLOCK_DISTRIBUTION_H
#ifdef __cplusplus
extern "C" {
#endif
#include "graph_defs.h"

struct artsBlockDistribution {
  graph_sz_t num_vertices; // the complete number of vertices
  graph_sz_t num_edges; // the complete number of edges
  graph_sz_t block_sz;
  artsGuid_t * graphGuid;
};

typedef struct artsBlockDistribution arts_block_dist_t;

graph_sz_t getBlockSize(const arts_block_dist_t* const _dist);
graph_sz_t getNodeBlockSize(node_t _node, const arts_block_dist_t* const _dist);
void initBlockDistribution(arts_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m);
node_t getOwner(vertex v, const arts_block_dist_t* const _dist);
vertex nodeStart(node_t n, const arts_block_dist_t* const _dist);
vertex nodeEnd(node_t n, const arts_block_dist_t* const _dist);

local_index_t getLocalIndex(vertex v, const arts_block_dist_t* const _dist);

// Note : This is always for current rank
// TODO should remove local arg. Added this because linker was complaining
// about artsGlobalRankId. Too tired to debug the problem ....
vertex getVertexId(node_t local, local_index_t, const arts_block_dist_t* const _dist);

void initBlockDistributionWithCmdLineArgs(arts_block_dist_t* _dist,
                                          int argc, 
                                          char** argv);

void freeDistribution(arts_block_dist_t* _dist);

artsGuid_t* getGuidForVertex(vertex v,
                             const arts_block_dist_t* const _dist);

artsGuid_t* getGuidForCurrentNode(const 
                                  arts_block_dist_t* const _dist);
#ifdef __cplusplus
}
#endif

#endif


