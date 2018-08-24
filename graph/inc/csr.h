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
#ifndef HAGGLE_CSR_H
#define HAGGLE_CSR_H
#ifdef __cplusplus
extern "C" {
#endif

#include "graph_defs.h"
#include "block_distribution.h"
#include "artsEdgeVector.h"

#define MAXCHAR 4096

typedef struct {
  graph_sz_t num_local_vertices;
  graph_sz_t num_local_edges;
  vertex* row_indices;
  vertex* columns;
  arts_block_dist_t* distribution;
  void* data;
} csr_graph;


void initCSR(csr_graph* _csr, 
             graph_sz_t _localv,
             graph_sz_t _locale,
             arts_block_dist_t* _dist,
             artsEdgeVector* _edges,
             bool _sorted_by_src);

int loadGraphNoWeight(const char* _file,
                      csr_graph* _graph,
                      arts_block_dist_t* _dist,
                      bool _flip,
                      bool _ignore_self_loops);

int loadGraphNoWeightCsr(const char* _file,
                        csr_graph* _graph,
                        arts_block_dist_t* _dist,
                        bool _flip,
                        bool _ignore_self_loops);

void printLocalCSR(const csr_graph* _csr);

int loadGraphUsingCmdLineArgs(csr_graph* _graph,
                              arts_block_dist_t* _dist,
                              int argc, char** argv);
void freeCSR(csr_graph* _csr);

void getNeighbors(csr_graph* _csr,
                  vertex v,
                  vertex** _out,
                  graph_sz_t* _neighborcount);
#ifdef __cplusplus
}
#endif

#endif
