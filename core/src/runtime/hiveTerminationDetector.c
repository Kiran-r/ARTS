#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <stdatomic.h>
#include "hiveRT.h"

// Termination detection counts                                                  
static unsigned int activeCount = 0;
static unsigned int finishedCount = 0;
static unsigned int lastFinishedCount = 0;
static unsigned int totalActiveCount = 0;

// Termination detection phases
enum {PHASE_1, PHASE_2} phase = PHASE_1;

hiveGuid_t startTermGuid = NULL_GUID;
hiveGuid_t getTermCountGuid = NULL_GUID;
hiveGuid_t reductionOpGuid = NULL_GUID;
hiveGuid_t dbReduxValGuid = NULL_GUID;

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

hiveGuid_t reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  /*Once every rank has sent their counter value, perform a reduction*/
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  unsigned int nodeCount = hiveGetTotalNodes();
  unsigned int sum = 0;
  counterVal *values = depv; //TODO: verify that depv actually contains all the inputs from all the ranks
  unsigned int totalActiveCount = 0;
  unsigned int totalFinishedCount = 0;
  unsigned int totalCount = 0;
  for (u32 count = 0; count < nodeCount; count++) {
    totalActiveCount += values[count]->curActiveCount;
    totalFinishedCount += values[count]->curFinishedCount;
  }
  totalCount = totalActiveCount - totalFinishedCount;
    
  if (totalCount) {  // re-start first phase, since totalCount is not zero
    phase = PHASE_1;
    // dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
    // TODO: verify that we can do the following by reusing the guid for redux?
    reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 0, 0, hiveGetTotalNodes());
    /*Now tell every rank to send their counter value to reductionop as depv*/
    unsigned int numNodes = hiveGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      hiveEdtCreate(getTermCount, rank, 1, (unsigned int*)&reductionOpGuid, 0);
    }
  } else if (phase == PHASE_2 &&  lastFinishedCount == totalFinishedCount) {
    /*Signal termination*/
    //TODO:  graceful exit with result
    hiveShutdown();
  }
  else {
    //start second phase
    phase = PHASE_2;
    lastFinishedCount = totalFinishedCount;
  }
}

hiveGuid_t getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int curActive = __atomic_load(&activeCount);
  unsigned int curFinished = __atomic_load(&finishedCount);
  counterVal *counterValue;
  hiveGuid_t counterValGuid = hiveDbCreate((hiveGuid_t**)&counterValue, sizeof(counterValue), false);
  counterValue->curActiveCount = curActive; 
  counterValue->curFinishedCount =  curFinished;
  /*signal reductionOp EDT with the counter values for this rank*/
  hiveSignalEdt(paramv[0], counterValGuid, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);// TODO: verify this
}

hiveGuid_t startTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  /*kick-off termination detection once every locality finished initialization*/
  // dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
  reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  /*TODO: verify how to ensure input buffers are big enough to get right counts in the proper slot?*/
  hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 0, 0, hiveGetTotalNodes());
  /*Now tell every rank to send their counter value to reductionop as depv*/
  unsigned int numNodes = hiveGetTotalNodes();
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    hiveEdtCreate(getTermCount, rank, 1, (unsigned int*)&reductionOpGuid, 0);
  }
}

hiveGuid_t localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  // if (!workerId) {
  activeCount = 0;
  finishedCount = 0;
  // we signal that initialization is done for this rank
  // TODO: We just need to signal that we are done. Check whether its the correct way to do it.
  // param[0] contains the GUID for start termination EDT
  hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
  //}
}

void hiveDetectTermination() { //TODO: maybe pass the rank from where we would want to start termination detection
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  /* Since everyone will call in the function, start termination from rank 0, worker 0 */
  if (!nodeId && !workerId) {
    /*Reserve an EDT for starting termination*/
    startTermGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(startTermination, startTermGuid, 0, 0, hiveGetTotalNodes());
    for (unsigned int rank = 0; rank < numNodes; rank++) {
      /*initialize termination counter on all ranks*/
      hiveEdtCreate(localTerminationInit, rank, 1, (unsigned int*)&startTermGuid, 0);
    }
  }
}
