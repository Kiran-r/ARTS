#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
// TODO: insert proper include

hiveGuid_t relaxGuid = NULL_GUID;

hiveGuid_t dummytask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  incrementFinishedCount(numNodes);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {

}

void initPerWorker(unsigned int nodeId, int argc, char** argv)
{
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  unsigned int numNodes = hiveGetTotalNodes();
  if(!nodeId && !workerId) {
    hiveDetectTermination(); //TODO: Fix placement of this call.
    incrementActiveCount(numNodes);
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      hiveEdtCreate(dummytask, rank, 0, 0, 0); 
    }
  }
}


int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}
