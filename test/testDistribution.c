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
#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsGraph.h"
#include "artsGlobals.h"
#include <assert.h>

int main(int argc, char** argv) {

#ifdef NDEBUG
  PRINTF("[WARN] asserts are disabled. Verification will not run.\n");
#endif

  arts_block_dist_t dist;
  initBlockDistribution(&dist, 64, 32, 2);
  assert(dist.num_vertices == 64);
  assert(dist.num_ranks == 2);
  assert(dist.block_sz == 32);
  assert(getOwner(5, &dist) == 0);
  assert(getOwner(31, &dist) == 0);
  assert(getOwner(45, &dist) == 1);
  assert(nodeStart(0, &dist) == 0);
  assert(nodeEnd(0, &dist) == 31);
  assert(nodeStart(1, &dist) == 32);
  assert(nodeEnd(1, &dist) == 63);

  initBlockDistribution(&dist, 8, 5, 3);
  assert(dist.num_vertices == 8);
  assert(dist.num_ranks == 3);
  assert(dist.block_sz == 3);
  assert(getOwner(5, &dist) == 1);
  assert(getOwner(6, &dist) == 2);
  assert(getOwner(2, &dist) == 0);
  assert(nodeStart(0, &dist) == 0);
  assert(nodeEnd(0, &dist) == 2);
  assert(nodeStart(2, &dist) == 6);
  assert(nodeEnd(2, &dist) == 7);

}
