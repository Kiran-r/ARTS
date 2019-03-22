/*
 * hive_tMT.c
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

#define PT_CONTEXTS  // maintain contexts via PThreads

#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <inttypes.h>
#define __USE_GNU
#include <string.h>

#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsDeque.h"
#include "artsDbFunctions.h"
#include "artsEdtFunctions.h"
#include "artsRemoteFunctions.h"

#include "hive_tMT.h"

#define DPRINTF( ... )
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
#define ONLY_ONE_THREAD 
//while(!hiveTestStateOneLeft(localPool->alias_running) && artsThreadInfo.alive)

msi_t * _hive_tMT_msi = NULL; // tMT shared data structure
__thread unsigned int aliasId = 0;
__thread msi_t * localPool = NULL;

static inline artsTicket GenTicket() 
{
    artsTicket ticket;
    ticket.fields.rank = artsGlobalRankId;
    ticket.fields.unit = artsThreadInfo.groupId;
    ticket.fields.thread = aliasId;
    ticket.fields.valid = 1;
    DPRINTF("r: %u u: %u t: %u v: %u\n", (unsigned int)ticket.fields.rank, (unsigned int)ticket.fields.unit, (unsigned int)ticket.fields.thread, (unsigned int)ticket.fields.valid);
    return ticket;
}

static inline bool hiveAccessorState(volatile accst_t* all_states, unsigned int start, bool flipstate) 
{
    uint64_t uint64_tState = *all_states;
    uint64_t current_pos = 1UL << start;  
    bool bState = (uint64_tState & current_pos) ? true : false;

    if(flipstate) 
        artsAtomicFetchXOrU64(all_states, current_pos);
    return bState;
}

static inline unsigned int hiveNextCandidate(volatile accst_t* all_states) 
{
	return ffsll(*all_states);
}

static inline bool hiveTestStateEmpty(volatile accst_t* all_states) 
{
    return !*all_states;
}

static inline bool hiveTestStateOneLeft(volatile accst_t* all_states) 
{
    return *all_states && !(*all_states & (*all_states-1));
}

static inline void hivePutToWork(unsigned int rank, unsigned int unit, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        msi_t * pool = &_hive_tMT_msi[unit];
        hiveAccessorState(pool->alias_running, thread, true);
        if(avail) hiveAccessorState(pool->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_post(&pool->sem[thread]) == -1) { //Wake avail thread up
            PRINTF("FAILED SEMI POST %u %u\n", artsThreadInfo.groupId, aliasId);
//            exit(EXIT_FAILURE);
        }
    #endif
    }
}

static inline void hivePutToSleep(unsigned int rank, unsigned int unit, unsigned int thread, bool avail) 
{
    if(rank == artsGlobalRankId)
    {
        msi_t * pool = &_hive_tMT_msi[unit];
        hiveAccessorState(pool->alias_running, thread, true);
        if(avail) hiveAccessorState(pool->alias_avail, thread, true);

    #ifdef PT_CONTEXTS
        if(sem_wait(&pool->sem[thread]) == -1) {
            PRINTF("FAILED SEMI WAIT %u %u\n", artsThreadInfo.groupId, aliasId);
//            exit(EXIT_FAILURE);
        }
    #endif
    }
}

static void* aliasThreadLoop(void* arg) 
{
    
    
    tmask_t * tArgs = (tmask_t*)arg;

    //set thread local vars
    aliasId = tArgs->aliasId;
    memcpy(&artsThreadInfo, tArgs->tlToCopy, sizeof(struct artsRuntimePrivate));
    
    unsigned int unitId = artsThreadInfo.groupId;
    unsigned int numAT = artsNodeInfo.tMT;
    DPRINTF("Setting: %u\n", unitId*(numAT-1)+(aliasId-1));
    artsNodeInfo.tMTLocalSpin[unitId*(numAT-1)+(aliasId-1)] = &artsThreadInfo.alive;
    
    localPool = &_hive_tMT_msi[artsThreadInfo.groupId];

    if(tArgs->unit->pin)
    {
        DPRINTF("PINNING to %u:%u\n", artsThreadInfo.groupId, aliasId);
//        artsAbstractMachineModelPinThread(tArgs->unit->coreInfo);
    }

    hiveAccessorState(localPool->alias_running, aliasId, true);
    hiveAccessorState(localPool->alias_avail, aliasId, true);
    
    if(sem_post(&localPool->sem[0]) == -1) {// finished  mask copy
        PRINTF("FAILED SEMI INIT POST %u %u\n", artsThreadInfo.groupId, aliasId);
//        exit(EXIT_FAILURE);
    }
    
    artsAtomicSub(&localPool->startUpCount, 1);

    hivePutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true); //Toggle availability
    ONLY_ONE_THREAD;
    
    artsRuntimeLoop();
    
    sem_post(&localPool->sem[0]);
    artsAtomicSub(&localPool->shutDownCount, 1);
}

static inline void CreateContexts(struct threadMask * mask, struct artsRuntimePrivate * semiPrivate) 
{
#ifdef PT_CONTEXTS
    tmask_t tmask;
    pthread_attr_t attr;
    long pageSize = sysconf(_SC_PAGESIZE);
    size_t size = pageSize;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, size);

    //Init semiphores
    unsigned int numAT = artsNodeInfo.tMT;
    for(int i = 0; i < numAT; ++i) 
    {
        if(sem_init(&localPool->sem[i], 0, 0) == -1) {
            PRINTF("FAILED SEMI INIT %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
    }
    
    tmask.unit = mask;
    tmask.tlToCopy = semiPrivate;
    for(int i = 1; i < numAT; ++i) 
    {
        tmask.aliasId = i;
        if(pthread_create(&localPool->aliasThreads[i-1], &attr, &aliasThreadLoop, &tmask)) {
            PRINTF("FAILED ALIAS THREAD CREATION %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
        DPRINTF("Master %u: Waiting in thread creation %d\n", artsThreadInfo.groupId, i);
        if(sem_wait(&localPool->sem[0]) == -1) { // wait to finish mask copy
            PRINTF("FAILED SEMI INIT WAIT %u %u\n", artsThreadInfo.groupId, i);
//            exit(EXIT_FAILURE);
        }
    }
#endif
}

static inline void DestroyContexts() {
#ifdef PT_CONTEXTS
    unsigned int numAT = artsNodeInfo.tMT;
    DPRINTF("SHUTDOWN ALIAS %u: %u\n", artsThreadInfo.groupId, localPool->shutDownCount);
    while(localPool->shutDownCount) {
        for(unsigned int i=1; i<numAT; i++)
            sem_post(&localPool->sem[i]);
//        PRINTF("SHUTDOWN ALIAS %u: %u\n", artsThreadInfo.groupId, localPool->shutDownCount);
    }
    
    DPRINTF("ALIAS JOIN: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<numAT-1; i++)
        pthread_join(localPool->aliasThreads[i], NULL);
    
    DPRINTF("SEM DESTROY: %u\n", artsThreadInfo.groupId);
    for(unsigned int i=0; i<numAT; i++)
        sem_destroy(&localPool->sem[i]);
#endif
}

// RT visible functions
// COMMENT: MasterThread (MT) is the original thread
void hive_tMT_NodeInit(unsigned int numThreads)
{
    if(artsNodeInfo.tMT == 1)
    {
        PRINTF("Temporal multi-threading only running 1 thread per core.  To context switch tMT > 1\n");
        artsNodeInfo.tMT = 0;
    }
    
    if(artsNodeInfo.tMT > 64)
    {
        PRINTF("Temporal multi-threading can't run more than 64 threads per core\n");
        artsNodeInfo.tMT = 64;
    }
    
    if(artsNodeInfo.tMT)
    {
        _hive_tMT_msi = (msi_t *) artsCalloc(numThreads * sizeof(msi_t));
        artsNodeInfo.tMTLocalSpin = (volatile bool**) artsCalloc(sizeof(bool*) * numThreads * (artsNodeInfo.tMT-1));
    }
}

void hive_tMT_RuntimePrivateInit(struct threadMask* unit, struct artsRuntimePrivate * semiPrivate) 
{
    unsigned int numAT = artsNodeInfo.tMT;
    localPool = &_hive_tMT_msi[artsThreadInfo.groupId];
    localPool->aliasThreads = (pthread_t*) artsMalloc(sizeof (pthread_t) * (numAT-1));
    
    // FIXME: for now, we'll live with an bitmap array structure...
    // FIXME: convert SOA into AOS to avoid collisions
    localPool->alias_running = (accst_t*) artsCalloc(sizeof (accst_t));
    *localPool->alias_running = 1UL; // MT is running on thread 0
    
    localPool->alias_avail   = (accst_t*) artsCalloc(sizeof (accst_t));
    //More clever ways break for 64 alias
    //Start at 1 since MT is bit 0 and is running
    for(unsigned int i=1; i<numAT; i++)
        *localPool->alias_avail |= 1UL << i;
    
    localPool->wakeUpNext = 0;
    localPool->wakeQueue  = artsNewQueue();

    localPool->sem = (sem_t*) artsMalloc(sizeof (sem_t) * (numAT));
    
    localPool->startUpCount = localPool->shutDownCount = numAT-1;
    CreateContexts(unit, semiPrivate);
    
    while(localPool->startUpCount);
    ONLY_ONE_THREAD;
}

void hive_tMTRuntimeStop()
{
    if(artsNodeInfo.tMT)
    {
        for(unsigned int i=0; i<artsNodeInfo.workerThreadCount * (artsNodeInfo.tMT-1); i++)
            *artsNodeInfo.tMTLocalSpin[i] = false;
    }
}

void hive_tMTRuntimePrivateCleanup()
{
    if(artsNodeInfo.tMT)
        DestroyContexts();
}

void artsNextContext() 
{
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
        if(!cand)
            cand = dequeue(localPool->wakeQueue);
        if(cand)
        {
            cand--;
            hivePutToWork(artsGlobalRankId, artsThreadInfo.groupId, cand, false); //already blocked don't flip
        }
        else
        {
            cand = (aliasId + 1) % artsNodeInfo.tMT;
            hivePutToWork(artsGlobalRankId, artsThreadInfo.groupId, cand, true);  //available so flip
        }

        hivePutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true);
        ONLY_ONE_THREAD;
    }
}

void artsWakeUpContext()
{
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
        if(!cand)
            cand = dequeue(localPool->wakeQueue);
        if(cand)
        {
            cand--;
            hivePutToWork( artsGlobalRankId, artsThreadInfo.groupId, cand,    false);
            hivePutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, true);
            ONLY_ONE_THREAD;
        }
    }
}
// End of RT visible functions

bool artsContextSwitch(unsigned int waitCount) 
{
    PRINTF("CONTEXT SWITCH\n");
    if(artsNodeInfo.tMT && artsThreadInfo.alive)
    {
        volatile unsigned int * waitFlag = &localPool->ticket_counter[aliasId];
        artsAtomicAdd(waitFlag, waitCount);
        while(*waitFlag)
        {
            unsigned int cand = artsAtomicSwap(&localPool->wakeUpNext, 0);
            if(!cand)
                cand = dequeue(localPool->wakeQueue);
            if(!cand)
                cand = hiveNextCandidate(localPool->alias_avail);
            if(cand)
                cand--;
            else {
                cand = (aliasId + 1) % artsNodeInfo.tMT;
            }
            hivePutToWork( artsGlobalRankId, artsThreadInfo.groupId, cand,    true);
            hivePutToSleep(artsGlobalRankId, artsThreadInfo.groupId, aliasId, false); // do not change availability
            ONLY_ONE_THREAD;
        }
        return true;
    }
    return false;
}

bool artsSignalContext(artsTicket_t waitTicket)
{
    PRINTF("SIGNAL CONTEXT\n");
    artsTicket ticket = (artsTicket) waitTicket;
    unsigned int rank   = (unsigned int)ticket.fields.rank;
    unsigned int unit   = (unsigned int)ticket.fields.unit;
    unsigned int thread = (unsigned int)ticket.fields.thread;
    
    if(artsNodeInfo.tMT)
    {
        if(ticket.bits)
        {
            if(rank == artsGlobalRankId)
            {
                if(!artsAtomicSub(&_hive_tMT_msi[unit].ticket_counter[thread], 1))
                {
                    unsigned int alias = thread + 1;
                    if(artsAtomicCswap(&_hive_tMT_msi[unit].wakeUpNext, 0, alias) != 0)
                        enqueue(alias, _hive_tMT_msi[unit].wakeQueue);
                }
            }
            else
            {
                artsRemoteSignalContext(rank, waitTicket);
            }
            return true;
        }
    }
    return false;
}

bool artsAvailContext()
{
    if(artsNodeInfo.tMT)
    {
        unsigned int cand = hiveNextCandidate(localPool->alias_avail);
        DPRINTF("R: %p A: %p Cand: %u\n", localPool->alias_running, localPool->alias_avail, cand);
        return cand != 0;
    }
    return false;
}

artsTicket_t artsGetContextTicket()
{
    artsTicket ticket;
    if(artsAvailContext())
        ticket = GenTicket();
    else
        ticket.bits = 0;
    DPRINTF("r: %u u: %u t: %u v: %u\n", (unsigned int)ticket.fields.rank, (unsigned int)ticket.fields.unit, (unsigned int)ticket.fields.thread, (unsigned int)ticket.fields.valid);
    return (artsTicket_t)ticket.bits;
}