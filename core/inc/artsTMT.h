/*
 * artsTMT
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

#ifndef CORE_INC_ARTS_TMT_H_
#define CORE_INC_ARTS_TMT_H_

#include <semaphore.h>
#include "arts.h"
#include "artsConfig.h"
#include "artsAbstractMachineModel.h"
#include "artsRuntime.h"
#include "artsGlobals.h"

#define MAX_TOTAL_THREADS_PER_MAX 65536
#define MAX_THREADS_PER_MASTER 64

typedef uint64_t accst_t;  // accessor state

typedef union
{
    uint64_t bits: 64;
    struct __attribute__((packed))
    {
        uint32_t   rank:   31;
        uint16_t   unit:   16; 
        uint16_t thread:   16;
        uint8_t   valid:    1;
    } fields;
} artsTicket;


typedef struct internalMsi
{
    pthread_t               * aliasThreads;
    sem_t                   * sem;
    volatile bool          ** alive;
    volatile unsigned int     ticket_counter[MAX_THREADS_PER_MASTER]; //Fixed counters used to keep track of outstanding promises (we can only wait on one context at a time)
    volatile accst_t          alias_running; // FIXME: right data structure?
    volatile accst_t          alias_avail;
    volatile unsigned int     startUpCount;
    volatile unsigned int     shutDownCount;
    struct internalMsi *           next;
} internalMsi_t;

typedef struct msi
{
    internalMsi_t *           head;
    volatile unsigned int     blocked;
    volatile unsigned int     total;
    volatile unsigned int     wakeUpNext;
    artsQueue *               wakeQueue;
} msi_t __attribute__ ((aligned (64)));                        // master shared info

typedef struct
{
  uint32_t                    aliasId;  // alias id
  struct artsRuntimePrivate * tlToCopy; // we copy the master thread's TL
  internalMsi_t             * localInternal;
  sem_t                     * startUpSem;
} tmask_t; // per alias thread info

// RTS internal interface
void artsTMTNodeInit(unsigned int numThreads);
void artsTMTRuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate);
void artsTMTRuntimePrivateCleanup();
void artsTMTRuntimeStop();

bool artsAvailContext();
void artsNextContext();
void artsWakeUpContext();

#endif /* CORE_INC_ARTS_TMT_H_ */
