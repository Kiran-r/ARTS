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


#include <stdio.h>
#include <stdlib.h>
#include "artsRT.h"

// Test
#include <pthread.h>

#include "artsGlobals.h"
#include "hiveFutures.h"

// debugging support
unsigned int ext_threadID();
unsigned int ext_threadUNIT();

uint64_t start;

void fibForkFut(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    // unpack
    uint64_t* futures = (uint64_t*) paramv[1];
    ticket_t ticket = (ticket_t) paramv[2]; // need all tickets
    unsigned int num_start = paramv[3];
    unsigned int num = paramv[4];
    unsigned int id = paramv[5];
    
    // compute
    uint64_t temp;
    uint64_t FutureFib[2];

    if(num < 2) {
        temp = num;
//        PRINTF("UNIT:MT/AT %lu:%lu Fib Anchor Ticket! 0x%llx  id=%d value=%d parent 0x%llx\n",
//            ext_threadUNIT(), ext_threadID(), 0, id, num, ticket); fflush(stdout);

    } else {
        uint64_t  args[] = {(uint64_t) fibForkFut, (uint64_t) FutureFib, 2, num_start, num-1, 0}; // second argument: # futures
        ticket_t tickets[2];
        tickets[0] = hiveCreateFuture(sizeof(args)/sizeof(uint64_t), args);

//        PRINTF("UNIT:MT/AT %lu:%lu Fib Ticket0! 0x%llx  value=%d parent 0x%llx\n",
//            ext_threadUNIT(), ext_threadID(), tickets[0], num-1, ticket); fflush(stdout);

        args[2] = 2;
        args[4] = num-2;
        args[5] = 1;
        tickets[1] = hiveCreateFuture(sizeof(args)/sizeof(uint64_t), args);

//        PRINTF("UNIT:MT/AT %lu:%lu Fib Ticket1! 0x%llx  value=%d parent 0x%llx\n",
//            ext_threadUNIT(), ext_threadID(), tickets[1], num-2, ticket); fflush(stdout);

        hiveGetFutures(tickets, 2);

        temp = FutureFib[0] + FutureFib[1];

//        PRINTF("UNIT:MT/AT %lu:%lu Fib Finished Ticket! 0x%llx  id=%d value=%d parent 0x%llx\n",
//            ext_threadUNIT(), ext_threadID(), tickets[1], id, id?num-2:num-1, ticket); fflush(stdout);

    }

    // pack
    futures[id] = temp;

    if (num_start == num)
    {
        // hiveSignalEdt(guid, fib, slot, DB_MODE_SINGLE_VALUE);
//        PRINTF("fibForkFut: finished recursion\n");
        uint64_t time = artsGetTimeStamp() - start;
        PRINTF("Fib %u: %u %lu\n", num_start, temp, time); //fflush(stdout);
        artsShutdown();
    }

}


void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

uint64_t root_FutureFib[2];

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        unsigned int num = atoi(argv[1]);
        uint64_t  args[] = {(uint64_t) NULL, (uint64_t) root_FutureFib, 2, num, num, 0}; // REQUIRED arg list for solver second argument: # futures
        start = artsGetTimeStamp();
        artsGuid_t guid = artsEdtCreate(fibForkFut, 0, 6, args, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
