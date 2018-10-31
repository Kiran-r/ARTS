/*
 * hive_tMT.h
 *
 *  Created on: March 30, 2018
 *      Author: Andres Marquez (@awmm)
 *
 *
 * This file is subject to the license agreement located in the file LICENSE
 * and cannot be distributed without it. This notice cannot be
 * removed or modified.
 *
 *
 *
 */

#ifndef CORE_INC_HIVE_TMT_H_
#define CORE_INC_HIVE_TMT_H_

#include <semaphore.h>
#include "artsConfig.h"
#include "artsAbstractMachineModel.h"
#include "artsRuntime.h"
#include "artsGlobals.h"

#include "hiveFutures.h"

#define MAX_THREADS_PER_MASTER 64

// types
typedef uint64_t accst_t; // accessor state

typedef enum hive_tMT_ReasonHandover
{
  NoReason   = 0,
  FutureReq  = 1,
  FutureGen  = 2,
  EndEDT     = 3
} rho_t; //reason for hand-over

typedef struct
{
  size_t                    numAT;                                        // # of potential aliases
  pthread_t*                pthread;                                      // all info pertinent to pthreads
  struct threadMask*        unit;       		          // current active alias
  unsigned int              ticket_counter[MAX_THREADS_PER_MASTER];
  volatile uint16_t         ticket_serial;
  struct artsDeque*         promise_queue;
  volatile accst_t*         alias_running __attribute__ ((aligned (64))); // FIXME: right data structure?
  volatile accst_t*         alias_avail;                    		  // FIXME: right data structure?
  volatile unsigned int     queue_own;
  artsQueue *               wakeQueue;
  // ownership baton is '1'
} ti_t; // shared amongst MT and AT pool

typedef struct
{
  uint32_t unitID;
  ti_t*    threadpool_info;
  bool     isMT;
} tci_t; // common interface for MT, AT

typedef struct
{
  pthread_t          pthreadMT; // master thread
  struct threadMask* unit;      // master thread info @ hive RT level
  sem_t*             sem;       // semaphores to synch within the pool
  ti_t*              ti;
} msi_t;                        // master shared info



// RTS internal interface
void hive_tMT_RuntimePrivateInit(struct threadMask* unit, struct artsConfig* config, struct artsRuntimePrivate * semiPrivate);
void artsContextSwitch(unsigned int waitCount);
void artsNextContext();
bool availContext();
void setContextAvail(unsigned int context);
unsigned int getCurrentContext();
void wakeUpContext();
#endif /* CORE_INC_HIVE_TMT_H_ */
