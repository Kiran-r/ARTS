#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>
#include "hive.h"
#include "hiveMalloc.h"
#include "hiveEdgeVector.h"

#define INCREASE_SZ_BY 2

//comparators
int compareBySource(const void* e1, const void* e2) {
  edge* pe1 = (edge*)e1;
  edge* pe2 = (edge*)e2;

  if (pe1->source < pe2->source)
    return -1;
  else if (pe1->source == pe2->source)
    return 0;
  else
    return 1;
}

int compareBySourceAndTarget(const void* e1, const void* e2) {
  edge* pe1 = (edge*)e1;
  edge* pe2 = (edge*)e2;

  if (pe1->source < pe2->source)
    return -1;
  else if (pe1->source == pe2->source) {
    if (pe1->target < pe2->target)
      return -1;
    else if (pe1->target == pe2->target) 
      return 0;
    else
      return 1;
  } else
    return 1;
}

// end comparators

void initEdgeVector(hiveEdgeVector *v, graph_sz_t initialSize) {
  v->edge_array = hiveMalloc(initialSize * sizeof(edge));
  v->used = 0;
  v->size = initialSize;
}

void pushBackEdge(hiveEdgeVector *v, vertex s, 
                  vertex t,
                  edge_data_t d) {

  if (v->used == v->size) {
    v->size *= INCREASE_SZ_BY;
    void* new = hiveRealloc(v->edge_array, v->size * sizeof(edge));
    if (!new) {
      PRINTF("[ERROR] Unable to reallocate memory. Cannot continue\n.");
      assert(false);
      return; // if -NDEBUG
    } 

    v->edge_array = new;
  }
  
#ifdef HIVE_PRINT_DEBUG
  PRINTF("used:%" PRIu64 ", size: %" PRIu64 "\n", v->used, v->size);
#endif

  v->edge_array[v->used].source = s;
  v->edge_array[v->used].target = t;
  v->edge_array[v->used++].data = d;
}

void printEdgeVector(const hiveEdgeVector *v) {
  for (u64 i = 0; i < v->used; ++i) {
    //PRINTF("(%" PRIu64 ", %" PRIu64 ", %" PRIu64 ")", v->edge_array[i].source, v->edge_array[i].target, v->edge_array[i].data);
    PRINTF("(%" PRIu64 ", %" PRIu64 ")", v->edge_array[i].source, 
           v->edge_array[i].target);
  }
}

void freeEdgeVector(hiveEdgeVector *v) {
  hiveFree(v->edge_array);
  v->edge_array = NULL;
  v->used = 0; 
  v->size = 0;
}

void sortBySource(hiveEdgeVector *v) {
  qsort((void*)v->edge_array,
        v->used,
        sizeof(edge),
        compareBySource);
        
}

void sortBySourceAndTarget(hiveEdgeVector *v) {
  qsort((void*)v->edge_array,
        v->used,
        sizeof(edge),
        compareBySourceAndTarget);
        
}
