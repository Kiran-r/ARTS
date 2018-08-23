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
