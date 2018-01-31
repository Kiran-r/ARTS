#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
//#include <stdatomic.h>
#include "hiveRT.h"
#include "hiveTerminationDetection.h"
#include "hiveAtomics.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

// Termination detection counts                                                  
static volatile unsigned int activeCount = 0;
static volatile unsigned int finishedCount = 0;
static volatile unsigned int lastActiveCount = 0;
static volatile unsigned int lastFinishedCount = 0;

// Termination detection phases
enum {PHASE_1, PHASE_2} phase = PHASE_1;

// Exit edt info
hiveGuid_t terminationExitGuid = NULL_GUID;
unsigned int terminationExitSlot = 0;

void incrementActiveCount(unsigned int n) {
//  __atomic_fetch_add(&activeCount, n, __ATOMIC_RELAXED);
  DPRINTF("INC ACTIVE COUNT\n");
  hiveAtomicAdd(&activeCount, n);
}

void incrementFinishedCount(unsigned int n) {
//  __atomic_fetch_add(&finishedCount, n, __ATOMIC_RELAXED);
    DPRINTF("INC FINISH COUNT\n");
  hiveAtomicAdd(&finishedCount, n);
}

hiveGuid_t getTermCount(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  DPRINTF("In getTermcount  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  unsigned int curActive = activeCount;
  unsigned int curFinished = finishedCount;
//  __atomic_load(&activeCount, &curActive, __ATOMIC_RELAXED);
//  __atomic_load(&finishedCount, &curFinished, __ATOMIC_RELAXED); 
  hiveSignalEdt(paramv[0], curActive, hiveGetCurrentNode(), DB_MODE_SINGLE_VALUE);
  hiveSignalEdt(paramv[0], curFinished, hiveGetCurrentNode() + hiveGetTotalNodes(), DB_MODE_SINGLE_VALUE);
}

hiveGuid_t reductionOp(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    DPRINTF("In reductionOp  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
    unsigned int numNodes = hiveGetTotalNodes();
    
    //Sum the active across nodes
    u64 totalActive = 0;
    for (unsigned int i = 0; i < numNodes; i++) 
        totalActive += depv[i].guid;
    
    //Sum the finished across nodes
    u64 totalFinish = 0;
    for (unsigned int i = numNodes; i < depc; i++) 
        totalFinish += depv[i].guid;
    
    u64 diff = totalActive - totalFinish;
    DPRINTF("Diff: %lu\n", diff);
    //We have a zero
    if(!diff) 
    {
        //Lets check the phase and if we have the same counts as before
        if(phase == PHASE_2 && lastActiveCount == totalActive && lastFinishedCount == totalFinish) 
        {
            DPRINTF("Calling finalization continuation provided by the user\n");
            hiveSignalEdt(terminationExitGuid, NULL_GUID, terminationExitSlot, DB_MODE_SINGLE_VALUE);
            return NULL_GUID; //
        }
        else //We didn't match the last one so lets try again
        {
            lastActiveCount = totalActive;
            lastFinishedCount = totalFinish;
            phase = PHASE_2;
            DPRINTF("Starting phase 2\n");
        }
    }
    else
        phase = PHASE_1;
  
    //We need to re-collect the sums again...
    hiveGuid_t reductionOpGuid = hiveEdtCreate(reductionOp, 0, 0, NULL, numNodes*2);
    for (unsigned int rank = 0; rank < numNodes; rank++) 
        hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
}

hiveGuid_t startTermination(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  /*kick-off termination detection once every locality finished initialization*/
  DPRINTF("In start termination count on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  unsigned int numNodes = hiveGetTotalNodes();
  hiveGuid_t reductionOpGuid = hiveEdtCreate(reductionOp, 0, 0, NULL, numNodes*2);
  /*Now tell every rank to send their counter value*/
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    hiveEdtCreate(getTermCount, rank, 1, (u64*)&reductionOpGuid, 0);
  }
}

//accept a guid. signal this guid when done
void hiveDetectTermination(hiveGuid_t finishGuid, unsigned int slot) { 
  DPRINTF("In hive detect termination  on node %u worker %u\n", hiveGetCurrentNode(), hiveGetCurrentWorker());
  /* Since everyone will call in the function, start termination from rank 0*/
  hiveGuid_t startTerminationGuid = hiveEdtCreate(startTermination, 0, 1, (u64*)&finishGuid, 0);
  terminationExitGuid = finishGuid;
  terminationExitSlot = slot;
}

hiveGuid_t localTerminationInit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
  activeCount = 0;
  finishedCount = 0;
  if (!hiveGetCurrentNode()) {
    lastActiveCount = 0;
    lastFinishedCount = 0;
    phase = PHASE_1;
  }
  DPRINTF("Finished initialization on rank: %u \n", hiveGetCurrentNode());
  // we signal that initialization is done for this rank
   hiveSignalEdt(paramv[0], 0, 0, DB_MODE_SINGLE_VALUE);
}

void initializeTerminationDetection(hiveGuid_t kickoffTerminationGuid) {
  unsigned int numNodes = hiveGetTotalNodes();
  for (unsigned int rank = 0; rank < numNodes; rank++) {
    /*initialize termination counter on all ranks*/
      hiveEdtCreate(localTerminationInit, rank, 1, (u64*)&kickoffTerminationGuid, 0);
  }
}
