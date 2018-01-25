#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
//#include <stdatomic.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"
#include "hiveAtomics.h"

// Termination detection counts                                                  
static volatile unsigned int activeCount = 0;
static volatile unsigned int finishedCount = 0;
static volatile unsigned int lastFinishedCount = 0;
unsigned int terminationExitSlot = 0;

volatile unsigned int totalActiveCount = 0;
volatile unsigned int totalFinishedCount = 0;

// Termination detection phases
enum {PHASE_1, PHASE_2} phase = PHASE_1;

hiveGuid_t startTerminationGuid = NULL_GUID;
hiveGuid_t getTermCountGuid = NULL_GUID;
hiveGuid_t reductionOpGuid = NULL_GUID;
hiveGuid_t dbReduxValGuid = NULL_GUID;
hiveGuid_t terminationExitGuid = NULL_GUID;
hiveGuid_t doneTerminationInitGuid = NULL_GUID;

typedef  struct {
  unsigned int curActiveCount; 
  unsigned int curFinishedCount;
} counterVal;

void incrementActiveCount(unsigned int n) {
//  __atomic_fetch_add(&activeCount, n, __ATOMIC_RELAXED);
  hiveAtomicAdd(&activeCount, n);
}

void incrementFinishedCount(unsigned int n) {
//  __atomic_fetch_add(&finishedCount, n, __ATOMIC_RELAXED);
  hiveAtomicAdd(&finishedCount, n);
}

hiveGuid_t incrementTotalActiveCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
//    __atomic_fetch_add(&totalActiveCount, *paramv, __ATOMIC_RELAXED);
    hiveAtomicAdd(&totalActiveCount, *paramv);
}

hiveGuid_t incrementTotalFinishedCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
//    __atomic_fetch_add(&totalFinishedCount, *paramv, __ATOMIC_RELAXED);
    hiveAtomicAdd(&totalFinishedCount, *paramv);
}

hiveGuid_t getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  printf("In getTermcount  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  unsigned int curActive = activeCount;
  unsigned int curFinished = curFinished;
//  __atomic_load(&activeCount, &curActive, __ATOMIC_RELAXED);
//  __atomic_load(&finishedCount, &curFinished, __ATOMIC_RELAXED);
  /*signal reductionOp EDT with the counter values for this rank*/
  hiveEdtCreate(incrementTotalActiveCount, 0, 1, (u64*)&curActive, 0);
  hiveEdtCreate(incrementTotalFinishedCount, 0, 1, (u64*)&curFinished, 0); 
  hiveSignalEdt(paramv[0], 0, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);
}

hiveGuid_t reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  printf("In reductionOp  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  /*Once every rank has sent their counter value, perform a reduction*/
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  unsigned int nodeCount = hiveGetTotalNodes();
  unsigned int sum = 0;
  unsigned int totalCount = 0;
  unsigned int curTotalActiveCount = totalActiveCount;
  unsigned int curTotalFinishedCount = totalFinishedCount;
//  __atomic_load(&totalActiveCount, &curTotalActiveCount, __ATOMIC_RELAXED);
//  __atomic_load(&totalFinishedCount, &curTotalFinishedCount, __ATOMIC_RELAXED);
  totalCount = curTotalActiveCount - curTotalFinishedCount;
  printf("Total count %u\n", totalCount);
  hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
  if (totalCount) {  // re-start first phase, since totalCount is not zero
    printf("Restarting Phase 1\n");
    phase = PHASE_1;
    reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (u64*)&paramv[0], hiveGetTotalNodes());
    /*Now tell every rank to send their counter value to reductionop as depv*/
    unsigned int numNodes = hiveGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
    }
  } else if (phase == PHASE_2 &&  lastFinishedCount == totalFinishedCount) {
    /*Signal termination*/
    printf("Calling finalization continuation provided by the user\n");
    hiveSignalEdt(terminationExitGuid, NULL_GUID, terminationExitSlot, DB_MODE_SINGLE_VALUE);
  }
  else {
    //start second phase
    printf("Starting phase 2\n");
    phase = PHASE_2;
    lastFinishedCount = curTotalFinishedCount;
    reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (u64*)&paramv[0], hiveGetTotalNodes());
    /*Now tell every rank to send their counter value to reductionop as depv*/
    unsigned int numNodes = hiveGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
    }
  }
}


hiveGuid_t startTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  /*kick-off termination detection once every locality finished initialization*/
  printf("In start termination count on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (u64*)&paramv[0], hiveGetTotalNodes());
  /*Now tell every rank to send their counter value*/
  unsigned int numNodes = hiveGetTotalNodes();
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
  }
}

hiveGuid_t localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  activeCount = 0;
  finishedCount = 0;
  printf("Finished initialization on rank: %u \n", nodeId);
  // we signal that initialization is done for this rank
   hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
}

void initializeTerminationDetection(hiveGuid_t kickoffTerminationGuid) {
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  if (!nodeId && !workerId) {
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      /*initialize termination counter on all ranks*/
      hiveEdtCreate(localTerminationInit, rank, 1, (u64*)&kickoffTerminationGuid, 0);
    }
  }
}

//accept a guid. signal this guid when done
void hiveDetectTermination(hiveGuid_t finishGuid, unsigned int slot) { 
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  
  printf("In hive detect termination  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  /* Since everyone will call in the function, start termination from rank 0*/
  startTerminationGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  hiveEdtCreate(startTermination, 0, 1, (u64*)&finishGuid, 0);
  terminationExitGuid = finishGuid;
  terminationExitSlot = slot;
}
