//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include "block_distribution.h"
#include <string.h>
#include <assert.h>
#include <inttypes.h>

graph_sz_t getBlockSize(const arts_block_dist_t* _dist) {
  graph_sz_t rem = _dist->num_vertices % artsGetTotalNodes();
  if (!rem)
    return (_dist->num_vertices /  artsGetTotalNodes());
  else
    return ((graph_sz_t)(_dist->num_vertices /  artsGetTotalNodes()) + 1);
}


void initBlockDistribution(arts_block_dist_t* _dist,
                           graph_sz_t _n,
                           graph_sz_t _m) {
  _dist->num_vertices = _n;
  _dist->num_edges = _m;
  _dist->block_sz = getBlockSize(_dist);

  // copied from ssspStart.c
  _dist->graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
  for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
    _dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i % artsGetTotalNodes());
  }
}

void initBlockDistributionWithCmdLineArgs(arts_block_dist_t* _dist,
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
  _dist->graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
  for(unsigned int i=0; i<artsGetTotalNodes(); i++) {
    _dist->graphGuid[i] = artsReserveGuidRoute(ARTS_DB_PIN, i % artsGetTotalNodes());
  }
}

void freeDistribution(arts_block_dist_t* _dist) {
  artsFree(_dist->graphGuid);
  _dist->graphGuid = NULL;

  _dist->num_vertices = 0;
  _dist->num_edges = 0;
  _dist->block_sz = 0;
}

artsGuid_t* getGuidForVertex(vertex v,
                             const arts_block_dist_t* const _dist) {
  node_t owner = getOwner(v, _dist);
  assert(owner < artsGetTotalNodes());
  return &(_dist->graphGuid[owner]);
}

artsGuid_t* getGuidForCurrentNode(const arts_block_dist_t* const _dist) {
  return &(_dist->graphGuid[artsGetCurrentNode()]);
}

node_t getOwner(vertex v, const arts_block_dist_t* const _dist) {
  return (node_t)(v / _dist->block_sz);
}

vertex nodeStart(node_t n, const arts_block_dist_t* const _dist) {
  return (vertex)((_dist->block_sz) * n);
}

vertex nodeEnd(node_t n, const arts_block_dist_t* const _dist) {
  // is this the last node ?
  if (n == (artsGetTotalNodes()-1)) {
    return (vertex)(_dist->num_vertices - 1);
  } else {
    return (nodeStart(n, _dist) + (_dist->block_sz-1));
  }
}

graph_sz_t getNodeBlockSize(node_t n, const arts_block_dist_t* const _dist) {
  // is this the last node
  if (n == (artsGetTotalNodes()-1)) {
    return (_dist->num_vertices - ((artsGetTotalNodes()-1)*_dist->block_sz));
  } else
    return _dist->block_sz;
}

local_index_t getLocalIndex(vertex v, 
                            const arts_block_dist_t* const _dist) {
  node_t n = getOwner(v, _dist);
  vertex base = nodeStart(n, _dist);
  assert(base <= v);
  return (v - base);
}

vertex getVertexId(node_t local_rank,
                   local_index_t u, const arts_block_dist_t* const _dist) {
  vertex v = nodeStart(local_rank, _dist);
  return (v+u);
}
