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
#include "arts.h"
#include "artsConfig.h"
#include "artsAbstractMachineModel.h"
#include "artsRuntime.h"
#include "artsGlobals.h"

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

typedef uint64_t artsTicket_t;

typedef struct
{
    pthread_t *               aliasThreads;                           //Actual pthreads
    sem_t *                   sem;                                    //Semaphores used to wake/sleep
    volatile unsigned int     ticket_counter[MAX_THREADS_PER_MASTER]; //Fixed counters used to keep track of outstanding promises (we can only wait on one context at a time)
    volatile accst_t *        alias_running; // FIXME: right data structure?
    volatile accst_t *        alias_avail;
    volatile unsigned int     wakeUpNext;
    artsQueue *               wakeQueue;
    volatile unsigned int     startUpCount;
    volatile unsigned int     shutDownCount;
} msi_t __attribute__ ((aligned (64)));                        // master shared info

typedef struct
{
  uint32_t                    aliasId;  // alias id
  struct threadMask *         unit;     // alias shared pool info
  struct artsRuntimePrivate * tlToCopy; // we copy the master thread's TL
} tmask_t; // per alias thread info

// RTS internal interface
void hive_tMT_NodeInit(unsigned int numThreads);
void hive_tMT_RuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate);
void hive_tMTRuntimePrivateCleanup();
void hive_tMTRuntimeStop();

void artsNextContext();
void artsWakeUpContext();

bool artsContextSwitch(unsigned int waitCount);
bool artsAvailContext();
bool artsSignalContext(artsTicket_t ticket);
artsTicket_t artsGetContextTicket();

#endif /* CORE_INC_HIVE_TMT_H_ */
