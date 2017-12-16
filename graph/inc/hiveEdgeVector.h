#ifndef HIVE_EDGE_VECTOR_H
#define HIVE_EDGE_VECTOR_H
#include "graph_defs.h"

#define EDGE_VEC_SZ 10000

typedef struct {
  edge *edge_array;
  graph_sz_t used;
  graph_sz_t size;
} hiveEdgeVector;

void initEdgeVector(hiveEdgeVector *v, graph_sz_t initialSize);
void pushBackEdge(hiveEdgeVector *v, vertex s, 
                        vertex t,
                        edge_data_t d);
void freeEdgeVector(hiveEdgeVector *v);
void sortBySource(hiveEdgeVector *v);
void sortBySourceAndTarget(hiveEdgeVector *v);
void printEdgeVector(const hiveEdgeVector *v);
#endif
