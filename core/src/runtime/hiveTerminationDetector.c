#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"


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

void terminateAsyncExecution(u32 paramc, u64 * paramv, u32 depc, 
			     hiveEdtDep_t depv[]) {
  
}


void reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  if (!nodeId && !workerId) {
    unsigned int sum = 0;
    unsigned int nodeCount = hiveGetTotalNodes();
    counterVal *values = depv;
    unsigned int totalActiveCount = 0;
    unsigned int totalFinishedCount = 0;
    unsigned int totalCount = 0;
    for (u32 count = 0; count < nodeCount; count++) {
      totalActiveCount += values[count]->curActiveCount;
      totalFinishedCount += values[count]->curFinishedCount;
    }
    totalCount = totalActiveCount - totalFinishedCount;
    
    if (totalCount) {  // re-start first phase
      phase = PHASE_1;
      dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
      reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
      /*pass in the dbguid to reduction, so that it can write the data there.*/
      hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (unsigned int*)&dbReduxValGuid, hiveGetTotalNodes());
      /*Now tell every rank to send their counter value to reductionop as depv*/
      hiveEdtCreate(getTermCount, nodeId, 1, (unsigned int*)&reductionOpGuid, sizeof(unsigned int) * 2 * hiveGetTotalNodes());
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
}

void getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int curActive = __atomic_load(&activeCount);
  unsigned int curFinished = __atomic_load(&finishedCount);
  counterVal *counterValue;
  hiveGuid_t counterValGuid = hiveDbCreate((void**)&counterValue, sizeof(counterValue), false);
  counterValue->curActiveCount = curActive; 
  counterValue->curFinishedCount =  curFinished;
  /*signal reductionOp EDT with the counter values for this rank*/
  hiveSignalEdt(paramv[0], counterValGuid, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);// TODO: verify this
}

void startTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  /*kick-off termination detection once every locality finished initialization*/
  //while(true) {
  dbReduxValGuid = hiveReserveGuidRoute(HIVE_DB, 0);  /*place where we would want to put the redux data*/
  reductionOpGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
  /*pass in the dbguid to reduction, so that it can write the data there.*/
  hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 1, (unsigned int*)&dbReduxValGuid, hiveGetTotalNodes());
  /*Now tell every rank to send their counter value to reductionop as depv*/
  hiveEdtCreate(getTermCount, nodeId, 1, (unsigned int*)&reductionOpGuid, sizeof(unsigned int) * 2 * hiveGetTotalNodes());
   
    //}
}

void localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  // if (!workerId) {
    activeCount = 0;
    finishedCount = 0;
    // we signal that initialization is done for this rank
    // TODO: We just need to signal that we are done. So instead of writing nodeId, is there a way to just signal the EDT?
    // param[0] contains the GUID for start termination EDT
    hiveSignalEdt(paramv[0], nodeId, nodeId, DB_MODE_SINGLE_VALUE);
    //}
}

void hiveDetectTermination() { //TODO: maybe pass the rank from where we would want to start termination detection
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  /* Since everyone will call in the function, start termination from rank 0, worker 0 */
  if(!nodeId && !workerId) {
    /*Reserve an EDT for starting termination*/
    startTermGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(startTermination, startTermGuid, 1, (unsigned int*)&getTermCountGuid, hiveGetTotalNodes());
    for (unsigned int loc = 0; loc < numNodes; loc++) {
      /*initialize termination counter on all ranks*/
      hiveEdtCreate(localTerminationInit, loc, 1, (unsigned int*)&startTermGuid, 0);
    }
  }
}
