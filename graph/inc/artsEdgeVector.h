#ifndef ARTS_EDGE_VECTOR_H
#define ARTS_EDGE_VECTOR_H
#ifdef __cplusplus
extern "C" {
#endif
#include "graph_defs.h"

#define EDGE_VEC_SZ 10000

typedef struct {
  edge *edge_array;
  graph_sz_t used;
  graph_sz_t size;
} artsEdgeVector;

void initEdgeVector(artsEdgeVector *v, graph_sz_t initialSize);
void pushBackEdge(artsEdgeVector *v, vertex s, 
                        vertex t,
                        edge_data_t d);
void freeEdgeVector(artsEdgeVector *v);
void sortBySource(artsEdgeVector *v);
void sortBySourceAndTarget(artsEdgeVector *v);
void printEdgeVector(const artsEdgeVector *v);
#ifdef __cplusplus
}
#endif

#endif
