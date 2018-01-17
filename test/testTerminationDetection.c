#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"


hiveGuid_t relaxGuid = NULL_GUID;
hiveGuid_t kickoffTerminationGuid = NULL_GUID;
hiveGuid_t exitProgramGuid = NULL_GUID;
hiveGuid_t doneGuid = NULL_GUID;

hiveGuid_t dummytask(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  incrementFinishedCount(1);
}

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]){
    hiveShutdown();
}

hiveGuid_t kickoffTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int numNodes = hiveGetTotalNodes();
  incrementActiveCount(numNodes);
  printf("Spawning tasks on different rank\n");
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    hiveEdtCreate(dummytask, rank, 0, 0, 0);
  }
  printf("kicking off termination detection  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  exitProgramGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  hiveEdtCreateWithGuid(exitProgram, exitProgramGuid, 0, NULL, 1);
  hiveDetectTermination(exitProgramGuid, 0); 
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {
}

void initPerWorker(unsigned int nId, int argc, char** argv)
{
  unsigned int workerId = hiveGetCurrentWorker();
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  if (!nodeId && !workerId) {
    kickoffTerminationGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(kickoffTermination, kickoffTerminationGuid, 0, NULL, hiveGetTotalNodes());
    printf("Created kickoffterminationguid on node %u worker %u\n", nodeId, workerId);
    initializeTerminationDetection(kickoffTerminationGuid);
  }
}


int main(int argc, char** argv)
{
  hiveRT(argc, argv);
  return 0;
}
