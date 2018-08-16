#include "block_distribution.h"
#include <string.h>
#include <assert.h>
#include <inttypes.h>

graph_sz_t getBlockSize(const hive_block_dist_t* _dist) {
  graph_sz_t rem = _dist->num_vertices % hiveGetTotalNodes();
  if (!rem)
    return (_dist->num_vertices /  hiveGetTotalNodes());
  else
    return ((graph_sz_t)(_dist->num_vertices /  hiveGetTotalNodes()) + 1);
}


void initBlockDistribution(hive_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m) {
  _dist->num_vertices = _n;
  _dist->num_edges = _m;
  _dist->block_sz = getBlockSize(_dist);

  // copied from ssspStart.c
  _dist->graphGuid = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
  for(unsigned int i=0; i<hiveGetTotalNodes(); i++) {
    _dist->graphGuid[i] = hiveReserveGuidRoute(HIVE_DB_PIN, i % hiveGetTotalNodes());
  }
}

void initBlockDistributionWithCmdLineArgs(hive_block_dist_t* _dist,
                                          int argc, 
                                          char** argv) {
  uint64_t n;
  uint64_t m;

  for (int i=0; i < argc; ++i) {
    if (strcmp("--num-vertices", argv[i]) == 0) {
      sscanf(argv[i+1], "%" SCNu64, &n);
    }

    if (strcmp("--num-edges", argv[i]) == 0) {
      sscanf(argv[i+1], "%" SCNu64, &m);
    }
  }

  PRINTF("[INFO] Initializing Block Distribution with following parameters ...\n");
  PRINTF("[INFO] Vertices : %" PRIu64 "\n", n);
  PRINTF("[INFO] Edges : %" PRIu64 "\n", m);

  _dist->num_vertices = n;
  _dist->num_edges = m;
  _dist->block_sz = getBlockSize(_dist);

  // copied from ssspStart.c
  _dist->graphGuid = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
  for(unsigned int i=0; i<hiveGetTotalNodes(); i++) {
    _dist->graphGuid[i] = hiveReserveGuidRoute(HIVE_DB_PIN, i % hiveGetTotalNodes());
  }
}

void freeDistribution(hive_block_dist_t* _dist) {
  hiveFree(_dist->graphGuid);
  _dist->graphGuid = NULL;

  _dist->num_vertices = 0;
  _dist->num_edges = 0;
  _dist->block_sz = 0;
}

hiveGuid_t* getGuidForVertex(vertex v,
                             const hive_block_dist_t* const _dist) {
  node_t owner = getOwner(v, _dist);
  assert(owner < hiveGetTotalNodes());
  return &(_dist->graphGuid[owner]);
}

hiveGuid_t* getGuidForCurrentNode(const hive_block_dist_t* const _dist) {
  return &(_dist->graphGuid[hiveGetCurrentNode()]);
}

node_t getOwner(vertex v, const hive_block_dist_t* const _dist) {
  return (node_t)(v / _dist->block_sz);
}

vertex nodeStart(node_t n, const hive_block_dist_t* const _dist) {
  return (vertex)((_dist->block_sz) * n);
}

vertex nodeEnd(node_t n, const hive_block_dist_t* const _dist) {
  // is this the last node ?
  if (n == (hiveGetTotalNodes()-1)) {
    return (vertex)(_dist->num_vertices - 1);
  } else {
    return (nodeStart(n, _dist) + (_dist->block_sz-1));
  }
}

graph_sz_t getNodeBlockSize(node_t n, const hive_block_dist_t* const _dist) {
  // is this the last node
  if (n == (hiveGetTotalNodes()-1)) {
    return (_dist->num_vertices - ((hiveGetTotalNodes()-1)*_dist->block_sz));
  } else
    return _dist->block_sz;
}

local_index_t getLocalIndex(vertex v, 
                            const hive_block_dist_t* const _dist) {
  node_t n = getOwner(v, _dist);
  vertex base = nodeStart(n, _dist);
  assert(base <= v);
  return (v - base);
}

vertex getVertexId(node_t local_rank,
                   local_index_t u, const hive_block_dist_t* const _dist) {
  vertex v = nodeStart(local_rank, _dist);
  return (v+u);
}
