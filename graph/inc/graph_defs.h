#ifndef ARTS_GRAPH_DEFS
#define ARTS_GRAPH_DEFS
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRT.h"
#include "artsGlobals.h"

typedef uint64_t vertex;
typedef uint64_t graph_sz_t;
typedef unsigned int node_t;
typedef uint32_t edge_data_t;
typedef uint64_t local_index_t;

typedef struct {
  vertex source;
  vertex target;
  edge_data_t data;
} edge;
#ifdef __cplusplus
}
#endif

#endif
