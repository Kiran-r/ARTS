#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"


/// Termination detection counts                                                                                                                                          
static unsigned int activeCount;
static unsigned int finishedCount;
unsigned int totalActiveCount;
hiveGuid_t startTermGuid = NULL_GUID;
hiveGuid_t getTermCountGuid = NULL_GUID;
hiveGuid_t reductionOpGuid = NULL_GUID;

//u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[], && hiveGetCurrentNode() == 0
void localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int nodeId = hiveGetCurrentNode();
  if (hiveGetCurrentWorker() == 0) {
    activeCount = 0;
    finishedCount = 0;
    //we signal that init is done for this node
    hiveSignalEdt(paramv[0], nodeId, nodeId, DB_MODE_SINGLE_VALUE);
  }
}

void reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  //TODO: probably have to loop through the dep array to accumulate the sum.
  for (u32 count = 0; count < depc; count++) {
    // depv[count] do something
  }
}

void getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  unsigned int curActive = __atomic_load(&activeCount);
  unsigned int curFinished = __atomic_load(&finishedCount);
  struct counterVal {unsigned int curActiveCount, unsigned int curFinishedCount};
  struct counterVal counterValue;
  counterValue.curActiveCount = curActive; 
  counterValue.curFinishedCount =  curFinished;
  hiveSignalEdt(paramv[0], counterValue, (hiveGetCurrentNode() - 1) * sizeof(unsigned int) * 2, DB_MODE_SINGLE_VALUE);// TODO: we know how many dependency there are but how to set data from each dependency?
}

void startTerminationEdt(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  while(true) {
    hiveEdtCreateWithGuid(reductionOp, reductionOpGuid, 0, 0, hiveGetTotalNodes());
    hiveGuid_t edtGuid = hiveEdtCreate(getTermCount, nodeId, 1, (unsigned int*)&reductionOpGuid, sizeof(unsigned int) * 2 * hiveGetTotalNodes());
   
  }
}

void detectTerminationEdt() {
  unsigned int numNodes = hiveGetTotalNodes();
  unsigned int nodeId = hiveGetCurrentNode();
  unsigned int workerId = hiveGetCurrentWorker();
  if(!nodeId) {
    startTermGuid = hiveReserveGuidRoute(HIVE_EDT, 0);
    hiveEdtCreateWithGuid(startTerminationEDT, startTermGuid, 1, (unsigned int*)&getTermCountGuid, hiveGetTotalNodes());
  } 
  if (!workerId) {
    for (unsigned int loc = 0; loc < numNodes; loc++) {
    hiveEdtCreate(localTerminationInit, loc, 1, &startTermGuid, 0);
    }
  }




 /* getTermCountGuid = hiveReserveGuidRoute(HIVE_EDT, 0); */
 /*    hiveEdtCreateWithGuid(getTermCount, getTermCountGuid, 0, 0, hiveGetTotalNodes()); */
 /*    hiveSignalEdt(, 1,  nodeId, DB_MODE_SINGLE_VALUE); */
 /*  // Tell everyone to initialize termination counters */
 /*  hiveEdtCreate(localTerminationInit, 0, 1, ) */
 /*  if(!nodeId) { */
 /*    getTermCountGuid = hiveReserveGuidRoute(HIVE_EDT, 0); */
    
  // TODO: How to make sure everyone is done with initialization?  
  // 1.If my rank is 0:                                            
  //  2a. Create a EDT to get a guid G with a value (termination_count_lco)
  //  3. create a bunch of broadcast EDTs with G as parameter.             
  //  4. Each of the broadcasted EDT will get back their values.          
  //  5. Once all the inputs are in, EDT created at step 2a will do a reduction on all input values.                                                                    
    // getTermCountGuid =  hiveMalloc(numNodes);
    //for (unsigned int loc=0; loc < numNodes; loc++) {

      //}
  }
//}
