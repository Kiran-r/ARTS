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
#ifndef ARTS_EDGE_VECTOR_H
#define ARTS_EDGE_VECTOR_H
#ifdef __cplusplus
extern "C" {
#endif
#include "graphDefs.h"

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
