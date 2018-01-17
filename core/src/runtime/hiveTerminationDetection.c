#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <stdatomic.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"

// Termination detection counts                                                  
static unsigned int activeCount = 0;
static unsigned int finishedCount = 0;
static unsigned int lastFinishedCount = 0;
//static unsigned int totalActiveCount = 0;
unsigned int  terminationExitSlot = 0;

unsigned int totalActiveCount = 0;
unsigned int totalFinishedCount = 0;

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
  __atomic_fetch_add(&activeCount, n, __ATOMIC_RELAXED);
}

void incrementFinishedCount(unsigned int n) {
  __atomic_fetch_add(&finishedCount, n, __ATOMIC_RELAXED);
}

/* hiveGuid_t terminateAsyncExecution(u32 paramc, u64 * paramv, u32 depc,  */
/* 			     hiveEdtDep_t depv[]) { */
/* } */


hiveGuid_t incrementTotalActiveCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    __atomic_fetch_add(&totalActiveCount, *paramv, __ATOMIC_RELAXED);
}

hiveGuid_t incrementTotalFinishedCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    __atomic_fetch_add(&totalFinishedCount, *paramv, __ATOMIC_RELAXED);
}

hiveGuid_t getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  printf("In getTermcount  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  unsigned int curActive, curFinished;
  __atomic_load(&activeCount, &curActive, __ATOMIC_RELAXED);
  __atomic_load(&finishedCount, &curFinished, __ATOMIC_RELAXED);
  /* counterVal *counterValue; */
  /* hiveGuid_t counterValGuid = hiveDbCreate((void**)&counterValue, sizeof(counterValue), false); */
  /* counterValue->curActiveCount = curActive;  */
  /* counterValue->curFinishedCount =  curFinished; */
  /* printf("Active and finished count being sent %u %u\n", counterValue->curActiveCount, counterValue->curFinishedCount); */
  /*signal reductionOp EDT with the counter values for this rank*/
  hiveEdtCreate(incrementTotalActiveCount, 0, 1, (u64*)&curActive, 0);
  hiveEdtCreate(incrementTotalFinishedCount, 0, 1, (u64*)&curFinished, 0); 
  hiveSignalEdt(paramv[0], 0, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);// TODO: verify this
}

hiveGuid_t reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  printf("In reductionOp  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  /*Once every rank has sent their counter value, perform a reduction*/
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  unsigned int nodeCount = hiveGetTotalNodes();
  unsigned int sum = 0;
  // counterVal *values = (counterVal *) depv; //TODO: verify that depv actually contains all the inputs from all the ranks
  /* unsigned int totalcurrentActiveCount = 0; */
  /* unsigned int totalFinishedCount = 0; */
  /* printf("Counter initialized to %u %u\n", totalActiveCount, totalFinishedCount); */
  unsigned int totalCount = 0;
  /* for (unsigned int count = 0; count < depc; count++) { */
  /*   counterVal *values = depv[count].ptr; */
  /*   // printf("Activecount %u\n", values[0].curActiveCount); */
  /*   /\* totalActiveCount += values->curActiveCount; *\/ */
  /*   /\* printf("Accumulated active count %u\n", totalActiveCount); *\/ */
  /*   /\* totalFinishedCount += values->curFinishedCount; *\/ */
  /*   /\* printf("Accumulated finish count %u\n", totalFinishedCount); *\/ */
  /* } */
  unsigned int curTotalActiveCount = 0;
  unsigned int curTotalFinishedCount = 0;
  __atomic_load(&totalActiveCount, &curTotalActiveCount, __ATOMIC_RELAXED);
  __atomic_load(&totalFinishedCount, &curTotalFinishedCount, __ATOMIC_RELAXED);
  totalCount = curTotalActiveCount - curTotalFinishedCount;
  printf("Total count %u\n", totalCount);
  hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
  if (totalCount) {  // re-start first phase, since totalCount is not zero
    printf("Restarting Phase 1\n");
    phase = PHASE_1;
    // dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
    // TODO: verify that we can do the following by reusing the guid for redux?
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
    // hiveShutdown();
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
  // dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
  reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  /*TODO: verify how to ensure input buffers are big enough to get right counts in the proper slot?*/
  //hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 0, NULL, hiveGetTotalNodes());
  hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (u64*)&paramv[0], hiveGetTotalNodes());
  /*Now tell every rank to send their counter value to reductionop as depv*/
  unsigned int numNodes = hiveGetTotalNodes();
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
  }
  // hiveShutDown();
  
  /* unsigned int numNodes = hiveGetTotalNodes(); */
  /* for (unsigned int rank = 0; rank < numNodes; rank++) { */
  /*   hiveEdtCreate(getTermCount, rank, 1, (u64*)&paramv[0], 0); */
  /* } */
  // hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
}

hiveGuid_t localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  printf("In local termination init\n");
  // if (!workerId) {
  activeCount = 0;
  finishedCount = 0;
  printf("Finished initialization on rank: %u \n", nodeId);
  // we signal that initialization is done for this rank
  // TODO: We just need to signal that we are done. Check whether its the correct way to do it.
  // param[0] contains the GUID for start termination EDT
  hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
  //}
}

hiveGuid_t doneTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {

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
void hiveDetectTermination(hiveGuid_t finishGuid, unsigned int slot) { //TODO: maybe pass the rank from where we would want to start termination detection
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  
  printf("In hive detect termination  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  /* Since everyone will call in the function, start termination from rank 0, worker 0 */
  //if (!nodeId && !workerId) {
    /*Reserve an EDT for starting termination*/
    startTerminationGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    // hiveEdtCreate(startTermination, 0, 0, NULL, 0);
    hiveEdtCreate(startTermination, 0, 1, (u64*)&finishGuid, 0);
    // hiveEdtCreateWithGuid(startTermination, startTerminationGuid, 0, 0, hiveGetTotalNodes());
    //}
  terminationExitGuid = finishGuid;
  terminationExitSlot = slot;
  // hiveSignalEdt(finishGuid, 0, 0, DB_MODE_SINGLE_VALUE);
}
