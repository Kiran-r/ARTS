#ifndef ARTS_GRAPH_DEFS
#define ARTS_GRAPH_DEFS
#ifdef __cplusplus
extern "C" {
#endif

#include "artsRT.h"
#include "artsGlobals.h"

typedef u64 vertex;
typedef u64 graph_sz_t;
typedef unsigned int node_t;
typedef u32 edge_data_t;
typedef u64 local_index_t;

typedef struct {
  vertex source;
  vertex target;
  edge_data_t data;
} edge;
#ifdef __cplusplus
}
#endif

#endif
