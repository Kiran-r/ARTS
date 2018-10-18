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

#define PT_CONTEXTS		// maintain contexts via PThreads

#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <inttypes.h>
#define	__USE_GNU
#include <string.h>

#include "artsAtomics.h"
#include "artsDeque.h"
#include "artsDbFunctions.h"
#include "artsEdtFunctions.h"
#include "arts.h"
#include "hive_tMT_impl.h"


msi_t*  _hive_tMT_msi=NULL; // tMT shared data structure

static struct artsConfig* _config=NULL; // configuration
static const unsigned int _idMT=0; // default master thread identifier
static const size_t _numBT=1; // # of BT (broker threads); currently BT not needed
static const size_t _numExpansions=1; // # of times to double queue sizes before abort;
static const size_t _numPromises=256; // # promises to track per MT (master thread)
static size_t _numAT; // # aliases; after initialization via configuration
static size_t _numAT_MT; // # aliases + master thread; after initialization via configuration

static __thread tmask_t tls; // thread local storage for AT (alias thread)
__thread tci_t tci; // common interface for MT & AT


// function declarations in support of USER interface
static inline bool isMT()
{
	return tci.isMT;
}


static inline unsigned int unitID()
{
	return tci.unitID;
}

uint32_t ext_threadID()  // for debugging purposes
{
	return isMT()?_idMT:tls.threadpool_id;
}

static inline threadid_t threadID()
{
	return isMT()?_idMT:tls.threadpool_id;
}

uint32_t ext_threadUNIT()  // for debugging purposes
{
	return tci.unitID;
}


static inline threadunit_t threadUNIT()
{
	return tci.unitID;
}


static inline ticket_t GenTicket(numtickets_t num_tickets)
{
	// serial for ticket creation potentially subjected to races; doesn't really matter though
	return ((((uint64_t) num_tickets) << 56) | (((uint64_t) threadUNIT()) << 48)
			| ((uint64_t) tci.threadpool_info->ticket_serial++) << 16 | (uint64_t) threadID());
}

static inline threadid_t GetTID_Ticket(ticket_t ticket)
{
	return ticket & 0xffff;
}

static inline uint16_t GetSerial_Ticket(ticket_t ticket)
{
	return (ticket >> 16) & 0xffff;
}

static inline threadid_t GetUNIT_Ticket(ticket_t ticket)
{
	return (ticket >> 48) & 0xff;
}

static inline threadid_t GetNum_Ticket(ticket_t ticket)
{
	return (ticket >> 56) & 0xff;
}

static inline bool hiveAccessorState(volatile accst_t* all_states, unsigned int start, bool init, bool next, bool flipstate)
{
	static uint64_t current_pos;
	bool bState;
	uint64_t  uint64_tState = *all_states;

	if (init)
		current_pos = 1UL << start;

	bState = (uint64_tState & current_pos)?true:false;

	if (flipstate)
	{
//		*all_states = uint64_tState ^ current_pos;
		artsAtomicFetchXOrU64(all_states, current_pos);
	}

	if (next)
		current_pos <<= 1;

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


 static inline void hivePutToWork(ti_t* threadpool_info, threadid_t TID, bool avail)
{
	hiveAccessorState(threadpool_info->alias_running, TID, true, false, true);
	if (avail) hiveAccessorState(threadpool_info->alias_avail, TID, true, false, true);

#ifdef PT_CONTEXTS
	unsigned int unit = threadpool_info->unit->id;
	msi_t* mastershare_info = _hive_tMT_msi;

	if (sem_post(&mastershare_info[unit].sem[TID])) // wake avail thread up
		exit(EXIT_FAILURE);
#endif
}


static inline void hivePutToSleep(ti_t*	threadpool_info, threadid_t TID, bool avail)
{
	hiveAccessorState(threadpool_info->alias_running, TID, true, false, true);
	if (avail) hiveAccessorState(threadpool_info->alias_avail, TID, true, false, true);

#ifdef PT_CONTEXTS
	unsigned int unit = threadpool_info->unit->id;
	msi_t* mastershare_info = _hive_tMT_msi;

	if (sem_wait(&mastershare_info[unit].sem[TID]))
		exit(EXIT_FAILURE);
#endif
}



static inline bool hiveRunPromise(void *edtPacket)
{ // return 'true' iff all required futures are generated

	struct artsEdt *edt = edtPacket;
    uint32_t depc = edt->depc;
    artsEdtDep_t* depv = (artsEdtDep_t *)(((uint64_t *)(edt + 1)) + edt->paramc);

    artsEdt_t func = edt->funcPtr;
    uint32_t paramc = edt->paramc;
    uint64_t *paramv = (uint64_t *)(edt + 1);

    func(paramc, paramv, depc, depv);
    bool allfutures = (bool) paramv[0];

    //	PRINTF("Promise Ticket! 0x%llx\n", paramv[2]); fflush(stdout);
    artsFree(edtPacket);

    return allfutures;
}

// alias threads
static void* AThreadLoop(void* arg)
{

  //--------- initialization
  tls = *(tmask_t*) arg;

  msi_t* mastershare_info = _hive_tMT_msi;
  ti_t*	 threadpool_info  = tls.threadpool_info;

  tci.isMT = false;
  tci.unitID = threadpool_info->unit->id;
  tci.threadpool_info = threadpool_info;

  unsigned int unit = threadUNIT();
  unsigned int TID = threadID();

  // we're officially running, not waiting and not available for scheduling
  hiveAccessorState(threadpool_info->alias_running, TID, true, false, true);
  hiveAccessorState(threadpool_info->alias_avail, TID, true, false, true);

  // PRINTF("AThreadLoop: UNIT: %d:%d, mastershare_info 0x%x\n", unit, threadID(),
  //		  	  mastershare_info); fflush(stdout);


  if (sem_post(&mastershare_info[unit].sem[_idMT])) // finished  mask copy
    exit(EXIT_FAILURE);

  DPRINTF("Master %lu: AThread 0x%x with threadpool_id %lu starting thread loop. Waiting for recruitment...\n",
		  tls.threadpool_info->unit->id, pthread_self(), threadID()); fflush(NULL);

  hivePutToSleep(threadpool_info, threadID(), true); // toggle availability

  while (1)
  {
	DPRINTF("UNIT:MT/AT %lu:%lu AThread recruited...\n", threadUNIT(), threadID()); fflush(NULL);

    struct artsEdt* edtFound;
	while(edtFound = artsDequePopFront(threadpool_info->promise_queue))
	{
		// assert(edtFound == artsDequePopFront(threadpool_info->promise_queue)); // Test pop twice to check

		if (hiveRunPromise(edtFound)) // put to sleep iff ancestor is woken up
			hivePutToSleep(threadpool_info, threadID(), true); // toggle availability

    }
  }
}


static inline void CreateContexts(ti_t*  threadpool_info)
{
#ifdef PT_CONTEXTS
  unsigned int unitID = threadpool_info->unit->id;
  tmask_t tmask;
  pthread_attr_t attr;
  long pageSize = sysconf(_SC_PAGESIZE);
  size_t size = pageSize;

  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, size);

  for (int i=0; i<_numAT_MT; ++i)
  {
	if (sem_init(&_hive_tMT_msi[unitID].sem[i], 0, 0))
		exit(EXIT_FAILURE);
  }


  tmask.threadpool_info = threadpool_info;
  threadpool_info->pthread[0] = (pthread_t) 0; // BT place holder; BT currently not needed
  for (int i=1; i<_numAT; ++i)
  { // create alias thread (ATs) pool and potentially BT
	tmask.threadpool_id = i;

	if (pthread_create(&threadpool_info->pthread[i], &attr, &AThreadLoop, &tmask))
    {
      exit(EXIT_FAILURE);
    }

	DPRINTF("Master %lu MThread 0x%x: Waiting in thread creation %d\n",
			threadpool_info->unit->id, pthread_self(), i); fflush(NULL);


	sem_wait(&_hive_tMT_msi[unitID].sem[_idMT]); // wait to finish mask copy

	DPRINTF("Master %lu MThread 0x%x: Done with thread creation %d\n",
			threadpool_info->unit->id, pthread_self(), i); fflush(NULL);

  }
#endif
}

static inline bool TicketManagement(ticket_t ticket)
{ // return 'true' iff ticket counter reaches '0'

	// FutureGen maintenance
	unsigned int num = GetNum_Ticket(ticket);
	threadunit_t unitID = GetUNIT_Ticket(ticket);    // client
	threadid_t   TID = GetTID_Ticket(ticket);        // client
	ti_t*        tpi = _hive_tMT_msi[unitID].ti;     // client


	if (num == 1)
	{
		hivePutToWork(tpi, TID, false); // do not change availability
		return true;
	}

	if (artsAtomicCswap(&tpi->ticket_counter[TID], 0, num-1) != 0)
	{
		if (artsAtomicSub(&tpi->ticket_counter[TID], 1) == 0)
		{
			hivePutToWork(tpi, TID, false); // do not change availability
			return true;
		}
	}


/*   // not thread-safe
	if (tpi->ticket_counter[TID] == 0) // initialize   // FIXME: ticket_counter should be per ticket not TID
	{
		tpi->ticket_counter[TID] = num-1;
		PRINTF("UNIT:MT/AT %lu:%lu ticket initialized by %lu:%lu to %d\n",
				unitID, TID, threadUNIT(), threadID(), num-1); fflush(NULL);

	}
	else
	{
		--tpi->ticket_counter[TID];
		PRINTF("UNIT:MT/AT %lu:%lu ticket decremented by %lu:%lu to %d\n",
				unitID, TID, threadUNIT(), threadID(), tpi->ticket_counter[TID]); fflush(NULL);
	}

	if (tpi->ticket_counter[TID] == 0) // check
	{
		PRINTF("UNIT:MT/AT %lu:%lu Enabled by %lu:%lu\n",
				unitID, TID, threadUNIT(), threadID()); fflush(NULL);

		hivePutToWork(tpi, TID, false); // do not change availability
		return true;
	}
*/
return false;
}


static void WrapPromise(uint32_t paramc, uint64_t* paramv, uint32_t depc, artsEdtDep_t depv[])
{
	artsEdt_t func = (artsEdt_t) paramv[0];
	func(paramc, paramv, depc, depv);

    //	PRINTF("Promise Ticket! 0x%llx\n", paramv[2]); fflush(stdout);
    ticket_t  ticket  = paramv[2];
    bool allfutures = TicketManagement(ticket);
    paramv[0] = (uint64_t) allfutures;
}


static inline void hivePromiseCreate(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t* paramv, uint32_t depc)
{
	msi_t*   mastershare_info = _hive_tMT_msi;
	ti_t*	 threadpool_info  = tci.threadpool_info;
    unsigned int edtSpace = sizeof(struct artsEdt) + paramc * sizeof(uint64_t) + depc * sizeof(artsEdtDep_t);
    struct artsEdt* edt = (struct artsEdt*)artsCalloc(edtSpace);
    edt->header.type = ARTS_EDT;
    edt->header.size = edtSpace;
    if(edt)
    {
    	edt->funcPtr = WrapPromise;
        edt->depc = 0;				    // promises do not have external dependencies
        edt->paramc = paramc;
        edt->currentEdt = NULL_GUID;    // promises have no guids assigned; executed locally
        edt->depcNeeded = 0;            // promises have all they dependencies encapsulated

        memcpy((uint64_t*) (edt+1), paramv, sizeof(uint64_t) * paramc);
        artsDequePushFront(threadpool_info->promise_queue, edt, 0); // priority meaningless
    }
}

static inline ticket_t _hiveCreateFuture(uint32_t paramc, uint64_t* paramv) // first parameter is pointer to future; last chooses DB creation
{

	// generate future ticket
	ticket_t ticket = GenTicket(paramv[2]);

	artsEdt_t funcPtr = (artsEdt_t) paramv[0];
	paramv[2] = ticket;                  // caller num_tickets; now this parameter holds the ticket
	hivePromiseCreate(funcPtr, 0, paramc, paramv, 0);

	PRINTF("UNIT:MT/AT %lu:%lu Thread 0x%x Created Future....\n",
			threadUNIT(), threadID(), pthread_self()); fflush(NULL);

	// return future ticket
	return ticket;
}


// USER interface
// returns a ticket for the outstanding future
ticket_t hiveCreateFuture(uint32_t paramc, uint64_t* paramv) // first parameter is promise, second pointer to future
{
	return _hiveCreateFuture(paramc, paramv);
}


// TODO we need a function for "any" ticketing

// retrieve future with ticket
void hiveGetFutures(ticket_t* ticket, unsigned int num)
{
	// FutureReq maintenance
	ti_t*	 threadpool_info  = tci.threadpool_info;
    msi_t* mastershare_info = _hive_tMT_msi;
    threadid_t TID = GetTID_Ticket(*ticket);

// move to promise queuing 	BTMessage(FutureReq, threadID(), ticket, num);

    //	PRINTF("UNIT:MT/AT %lu:%lu Thread 0x%x Getting Futures....\n",
    //       threadUNIT(), threadID(), pthread_self()); fflush(NULL);

		unsigned int cand=0;
		if (hiveTestStateOneLeft(threadpool_info->alias_running)) // only current thread is running: draft!
		{
			assert(!hiveTestStateEmpty(threadpool_info->alias_avail));
			cand = hiveNextCandidate(threadpool_info->alias_avail); // WARN: if multi-entry, potential race condition

			PRINTF("b) Unit %lu: Candidate %d Alias_Running 0x%x\n", unitID(), cand, *threadpool_info->alias_running); fflush(NULL);

			if (cand)
			{
				unsigned int alias = cand-1;
				assert(alias != _idMT);
				assert(!hiveAccessorState(threadpool_info->alias_running, alias, true, false, false));
				hivePutToWork(threadpool_info, alias, true);

				if (hiveTestStateEmpty(threadpool_info->alias_avail)
						|| hiveTestStateOneLeft(threadpool_info->alias_avail))
					PRINTF("Unit %lu: GetFutures: WARNING: One or less alias left!\n",
								unitID(), pthread_self()); fflush(NULL);
			}
		}
		// put future requester to sleep
		hivePutToSleep(threadpool_info, threadID(), false); // do not change availability

}

 // retrieve future with ticket
void hiveGetFuture(ticket_t ticket)
{
	hiveGetFutures(&ticket, 1);
}
// End of USER visible functions


// RT visible functions

// COMMENT: MasterThread (MT) is the original thread
void hive_tMT_RuntimePrivateInit(struct threadMask* unit, struct artsConfig* config)
{ // run during master's initialization
         _numAT = config->tMT;
         _numAT_MT = _numAT+1;      // # aliases + master thread
         _config = config;
  size_t num_tMT_entries =_numExpansions * _numAT;

  assert(_numAT <= 62); // FIXME: currently max. number of alias threads

  unsigned int unitID = unit->id;

  ti_t*  threadpool_info = (ti_t*) artsCalloc(sizeof(ti_t)); // shared info amongst BrokerThread (BT) and AliasThread (AT)
  threadpool_info->unit =  unit; // master's info
  threadpool_info->numAT = _numAT;
  threadpool_info->current_alias_id = 0; // no aliasing at this point
  threadpool_info->pthread = (pthread_t*) artsMalloc(sizeof(pthread_t) * (_numBT + _numAT));
  // FIXME: convert SOA into AOS to avoid collisions
  threadpool_info->alias_running  = (accst_t*) artsCalloc(sizeof(accst_t));
  *threadpool_info->alias_running = 1UL << _idMT; // MT is running
  threadpool_info->alias_avail    = (accst_t*) artsCalloc(sizeof(accst_t));
  *threadpool_info->alias_avail   = (1UL << (_numAT_MT))-1UL; // FIXME: init bitmap
  *threadpool_info->alias_avail ^= 1UL << _idMT; // MT already being used
  threadpool_info->ticket_serial = 0;
  threadpool_info->promise_queue = artsDequeNew(_numPromises);
  threadpool_info->queue_own = 1; // queue open for business

  tci.isMT = true;
  tci.unitID = unitID;
  tci.threadpool_info = threadpool_info;

  // FIXME: for now, we'll live with an bitmap array structure...
  // MT is located at state bit 0
  assert(_numExpansions * (_numAT_MT) <= sizeof(accst_t)*8); // FIXME: _numExpansions not implemented

  // sync on _idMT (MT waits)
  _hive_tMT_msi[unitID].pthreadMT = pthread_self();
  _hive_tMT_msi[unitID].unit = unit; // master's info
  _hive_tMT_msi[unitID].sem = (sem_t*) artsMalloc(sizeof(sem_t) * (_numAT_MT));
  _hive_tMT_msi[unitID].ti = threadpool_info;


  CreateContexts(threadpool_info);

  }
// End of RT visible functions
