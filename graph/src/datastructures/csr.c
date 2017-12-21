#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include "csr.h"
#include "hiveEdgeVector.h"

//#define PRINTF printf

void initCSR(csr_graph* _csr, 
             graph_sz_t _localv,
             graph_sz_t _locale,
             hive_block_dist_t* _dist,
             hiveEdgeVector* _edges,
             bool _sorted_by_src) {

  // data is a single array that merges row_indices and columns
  graph_sz_t totsz = (_localv+1)+_locale;

#ifndef NO_HIVE_MALLOC
  _csr->data = hiveDbCreateWithGuid(*getGuidForCurrentNode(_dist), 
                                    totsz * sizeof(vertex), 
                                    true/*no moving*/);
#else
  _csr->data = malloc(totsz * sizeof(vertex));
#endif

  _csr->num_local_vertices = _localv;
  _csr->num_local_edges = _locale;
  _csr->distribution = _dist;
  _csr->row_indices = (vertex*)_csr->data;
  _csr->columns = (vertex*)(_csr->data + (_localv+1)*sizeof(vertex));

  for(u64 i=0; i <= _localv; ++i) {
    _csr->row_indices[i] = 0;
  }

  for(u64 i=0; i < _locale; ++i) {
    _csr->columns[i] = 0;
  }

#ifdef HIVE_PRINT_DEBUG
  printEdgeVector(_edges);
#endif
  //  PRINTF("\n4\n");
  // if _edges are not sorted, then sort them
  if (!_sorted_by_src) {
    sortBySource(_edges);
  }

#ifdef HIVE_PRINT_DEBUG
  printEdgeVector(_edges);
#endif

  //  PRINTF("5\n");
  vertex last_src = _edges->edge_array[0].source;
  vertex t = _edges->edge_array[0].target;
  vertex src_ind = getLocalIndex(last_src, _dist);

  _csr->columns[0] = t;
  _csr->row_indices[src_ind] = 0;
  _csr->row_indices[src_ind+1] = 1;

  //  PRINTF("Test=%" PRIu64 "\n",  _csr->columns[0]); 

  /*  PRINTF("\n");
      for (u64 t=0; t <= _localv; ++t) {
      PRINTF("B%" PRIu64 ", ",  _csr->row_indices[t]); 
      }
      PRINTF("\n");*/

  // populate edges
  for (u64 i=1; i < _edges->used; ++i) {
    vertex s = _edges->edge_array[i].source;
    vertex t = _edges->edge_array[i].target;
    src_ind = getLocalIndex(s, _dist);

    if (s == last_src) {
      // refers to previous source
      _csr->columns[i] = t;
      ++(_csr->row_indices[src_ind+1]);
    } else {
      // if there are vertices without edges, those indexes need to be
      // set
      vertex last_src_ind = getLocalIndex(last_src, _dist);
      ++last_src_ind;
      vertex val = _csr->row_indices[last_src_ind];
      assert(last_src_ind <= src_ind);
      while(last_src_ind != src_ind) {
        _csr->row_indices[++last_src_ind] = val;
      }

      // new source
      //assert(_csr->row_indices[src_ind] == i);
      //PRINTF("src_ind = %" PRIu64 ", ",  src_ind);
      _csr->row_indices[src_ind+1] = i+1;
      _csr->columns[i] = t;
      last_src = s;
    }
  }


  // initialize until the end of the vertex array
  vertex last_src_ind = getLocalIndex(last_src, _dist);
  ++last_src_ind;
  vertex val = _csr->row_indices[last_src_ind];
  assert(last_src_ind <= _localv);
  while(last_src_ind < _localv) {
    _csr->row_indices[++last_src_ind] = val;
  }

  /*PRINTF("Test00=%" PRIu64 "\n",  _csr->columns[0]); 

    PRINTF("\n");
    for (u64 t=0; t <= _localv; ++t) {
    PRINTF("%" PRIu64 ", ",  _csr->row_indices[t]); 
    }
    PRINTF("\n");

    PRINTF("\n");
    for (u64 p=0; p < _locale; ++p) {
    PRINTF("%" PRIu64 ", ",  _csr->columns[p]); 
    }
    PRINTF("\n");*/
}

void printLocalCSR(const csr_graph* _csr) {

  // print metadata
  PRINTF("\n=============================================\n");
  PRINTF("[INFO] Number of local vertices : %" PRIu64 "\n", 
         _csr->num_local_vertices);
  PRINTF("[INFO] Number of local edges : %" PRIu64 "\n", 
         _csr->num_local_edges);

  u64 i, j;
  u64 nedges = 0;
  for (i=0; i < _csr->num_local_vertices; ++i) {
    if (nedges == _csr->num_local_edges)
      break;

    vertex v = getVertexId(hiveGetCurrentNode(), i, _csr->distribution);

    for(j = _csr->row_indices[i]; j < _csr->row_indices[i+1]; ++j) {
      vertex u = _csr->columns[j];
      printf("(%" PRIu64 ", %" PRIu64 ")\n", v, u); 
      ++nedges;
    }
  }

  PRINTF("\n=============================================\n");

}

void freeCSR(csr_graph* _csr) {
#ifndef NO_HIVE_MALLOC
  hiveFree(_csr->data);
#else
  free(_csr->data);
#endif

  _csr->data = NULL;
  _csr->num_local_vertices = 0;
  _csr->num_local_edges = 0;
  _csr->distribution = NULL;
  _csr->row_indices = NULL;
  _csr->columns = NULL;
}

int loadGraphUsingCmdLineArgs(csr_graph* _graph,
                              hive_block_dist_t* _dist,
                              int argc, char** argv) {
  bool flip = false;
  bool keep_self_loops = false;
  char* file = NULL;

  for (int i=0; i < argc; ++i) {
    if (strcmp("--file", argv[i]) == 0) {
      file = argv[i+1];
    }

    if (strcmp("--flip", argv[i]) == 0) {
      flip = true;
    }

    if (strcmp("--keep-self-loops", argv[i]) == 0) {
      keep_self_loops = true;
    }
  }

  PRINTF("[INFO] Initializing GraphDB with following parameters ...\n");
  PRINTF("[INFO] Graph file : %s\n", file);
  PRINTF("[INFO] Flip ? : %d\n", flip);
  PRINTF("[INFO] Keep Self-loops ? : %d\n", keep_self_loops);

  return loadGraphNoWeight(file,
                           _graph,
                           _dist,
                           flip,
                           !keep_self_loops);
                           

}

void getNeighbors(csr_graph* _csr,
                  vertex v,
                  vertex** _out,
                  graph_sz_t* _neighborcount) {
  // make sure vertex belongs to current node
  assert(getOwner(v, _csr->distribution) 
         == hiveGetCurrentNode());

  // get the local index for the vertex
  local_index_t i = getLocalIndex(v, _csr->distribution);
  // get the column start position
  graph_sz_t start = _csr->row_indices[i];
  graph_sz_t end = _csr->row_indices[i+1];

  (*_out) = &(_csr->columns[start]);
  (*_neighborcount) = (end-start);
}

// If we want to read the graph as an undirected
// graph set _flip = True
int loadGraphNoWeight(const char* _file,
                      csr_graph* _graph,
                      hive_block_dist_t* _dist,
                      bool _flip,
                      bool _ignore_self_loops) {

  FILE *file = fopen(_file, "r");
  if (file == NULL) {
    PRINTF("[ERROR] File cannot be open -- %s", _file);
    return -1;
  } 
  
  hiveEdgeVector vedges;
  initEdgeVector(&vedges, EDGE_VEC_SZ);

  char str[MAXCHAR];
  while (fgets(str, MAXCHAR, file) != NULL) {
    if (str[0] == '%')
      continue;

    if (str[0] == '#')
      continue;

    // We do not know how many edges we are going to load
    graph_sz_t src, target;
    edge_data_t weight;

    char* token = strtok(str, "\t");
    int i = 0;
    while(token != NULL) {
      if (i == 0) { // source
        src = atoll(token);
        ++i;
      } else if (i == 1) { // target
        target = atoll(token);
        i = 0;
      }

      //printf("src=%llu, target=%llu\n", src, target);
      token = strtok(NULL, " ");
    }

    if (_ignore_self_loops && (src == target))
      continue;

    // TODO weights
    // source belongs to current node
    if (getOwner(src, _dist) == hiveGetCurrentNode()) {
      pushBackEdge(&vedges, src, target, 0/*weight zeor for the moment*/);
    } /*else {
      printf("src = %" PRIu64 ", owner = %d, global rank : %d", src,
             getOwner(src, _dist),
             hiveGetCurrentNode());
      assert(false); //TODO remove
      }*/

    if (_flip) {
      if (getOwner(target, _dist) == hiveGetCurrentNode()) {
        pushBackEdge(&vedges, target, src, 0/*weight zeor for the moment*/);
      } 
    }
  }

  fclose(file);

  // done loading edge -- sort them by source
  sortBySource(&vedges);

  initCSR(_graph, 
          getNodeBlockSize(hiveGetCurrentNode(), _dist),
          vedges.used,
          _dist,
          &vedges,
          true);

  freeEdgeVector(&vedges);

  return 0;
}

