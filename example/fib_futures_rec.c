///*
// * hive_tMT.h
// *
// *  Created on: March 30, 2018
// *      Author: Andres Marquez (@awmm)
// *
// *
// * This file is subject to the license agreement located in the file LICENSE
// * and cannot be distributed without it. This notice cannot be
// * removed or modified.
// *
// *
// *
// */
//
//
//#include <stdio.h>
//#include <stdlib.h>
//#include "hiveRT.h"
//
//// Test
//#include <pthread.h>
//
//#include "hiveGlobals.h"
//#include "hiveFutures.h"
//
//// debugging support
//unsigned int ext_threadID();
//unsigned int ext_threadUNIT();
//
//hiveGuid_t fibIterative(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//	hiveGuid_t   guid = paramv[0];
//    unsigned int slot = paramv[1];
//    unsigned int    n = paramv[2];
//
//	if (n < 2) {
//		hiveSignalEdt(guid, n, slot, DB_MODE_SINGLE_VALUE);
//	}
//
//	int fib = 1;
//	int prevFib = 1;
//
//	for(int i=2; i<n; i++) {
//		int temp = fib;
//		fib+= prevFib;
//		prevFib = temp;
//	}
//	hiveSignalEdt(guid, fib, slot, DB_MODE_SINGLE_VALUE);
//}
//
//
//hiveGuid_t promise(u32 paramc, u64* paramv, u32 depc, hiveEdtDep_t depv[])
//{
//	PRINTF("UNIT:MT/AT %lu:%lu Thread 0x%x: started promise1\n",
//			ext_threadUNIT(), ext_threadID(), pthread_self()); fflush(NULL);
//
//	// unpack
//	u64* future = (u64*) paramv[1];
//
////	PRINTF("promise fut[0]=%lu fut[1]=%lu\n", future[0], future[1]); fflush(NULL);
//
//	int fib = paramv[3];
//	int prevFib = paramv[4];
//
////	PRINTF("promise fib=%lu prevFib=%lu\n", fib, prevFib); fflush(NULL);
//
//
//	// compute
//	int temp = fib;
//	fib+= prevFib;
//	prevFib = temp;
//
//	// pack
//	future[0] = fib;
//	future[1] = prevFib;
//
//	PRINTF("finished promise\n"); fflush(stdout);
//}
//
//
//hiveGuid_t fibIterativeFut(u32 paramc, u64* paramv, u32 depc, hiveEdtDep_t depv[])
//{
//	hiveGuid_t   guid = paramv[0];
//    unsigned int slot = paramv[1];
//    unsigned int    n = paramv[2];
//
//	PRINTF("UNIT:MT/AT %lu:%lu Thread 0x%x: started fib\n",
//			ext_threadUNIT(), ext_threadID(), pthread_self()); fflush(NULL);
//
//
//	if (n < 2) {
//		hiveSignalEdt(guid, n, slot, DB_MODE_SINGLE_VALUE);
//	}
//
//	u64 fib = 1;
//	u64 prevFib = 1;
//	for(int i=2; i<n; i++) {
//		// u64* FutureFib;
//		u64  FutureFib[2];
//
//		u64  args[] = {(u64) promise, (u64) FutureFib, 1, fib, prevFib}; // second argument now defines num tickets
//		ticket_t ticket = hiveCreateFuture(sizeof(args)/sizeof(u64), args);
//
//		// overlap region
//
//		hiveGetFuture(ticket);
//		fib = FutureFib[0];
//		prevFib = FutureFib[1];
//	}
//
//	hiveSignalEdt(guid, fib, slot, DB_MODE_SINGLE_VALUE);
//}
//
//hiveGuid_t fibForkFut(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//	// unpack
//	u64* futures = (u64*) paramv[1];
//	ticket_t ticket = (ticket_t) paramv[2]; // need all tickets
//	unsigned int num_start = paramv[3];
//	unsigned int num = paramv[4];
//	unsigned int id = paramv[5];
//
//	// compute
//	u64 temp;
//	u64 FutureFib[2];
//
//	if(num < 2) {
//		temp = num;
//		PRINTF("UNIT:MT/AT %lu:%lu Fib Anchor Ticket! 0x%llx  id=%d value=%d parent 0x%llx\n",
//				ext_threadUNIT(), ext_threadID(), 0, id, num, ticket); fflush(stdout);
//
//	} else {
//		u64  args[] = {(u64) fibForkFut, (u64) FutureFib, 2, num_start, num-1, 0}; // second argument: # futures
//		ticket_t tickets[2];
//		tickets[0] = hiveCreateFuture(sizeof(args)/sizeof(u64), args);
//
//		PRINTF("UNIT:MT/AT %lu:%lu Fib Ticket0! 0x%llx  value=%d parent 0x%llx\n",
//				ext_threadUNIT(), ext_threadID(), tickets[0], num-1, ticket); fflush(stdout);
//
//		args[2] = 2;
//		args[4] = num-2;
//		args[5] = 1;
//		tickets[1] = hiveCreateFuture(sizeof(args)/sizeof(u64), args);
//
//		PRINTF("UNIT:MT/AT %lu:%lu Fib Ticket1! 0x%llx  value=%d parent 0x%llx\n",
//				ext_threadUNIT(), ext_threadID(), tickets[1], num-2, ticket); fflush(stdout);
//
//		hiveGetFutures(tickets, 2);
//
//		temp = FutureFib[0] + FutureFib[1];
//
//		PRINTF("UNIT:MT/AT %lu:%lu Fib Finished Ticket! 0x%llx  id=%d value=%d parent 0x%llx\n",
//				ext_threadUNIT(), ext_threadID(), tickets[1], id, id?num-2:num-1, ticket); fflush(stdout);
//
//	}
//
//	// pack
//	futures[id] = temp;
//
//	if (num_start == num)
//	{
//		// hiveSignalEdt(guid, fib, slot, DB_MODE_SINGLE_VALUE);
//		PRINTF("fibForkFut: finished recursion\n");
//		PRINTF("Fib %u: %u \n", num_start, temp); fflush(stdout);
//		hiveShutdown();
//	}
//
//}
//
//hiveGuid_t fibJoin(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//    unsigned int x = depv[0].guid;
//    unsigned int y = depv[1].guid;
//    hiveSignalEdt(paramv[0], x+y, paramv[1], DB_MODE_SINGLE_VALUE);
//}
//
//
//hiveGuid_t fibFork(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//    hiveGuid_t guid = paramv[0];
//    unsigned int slot = paramv[1];
//    unsigned int num = paramv[2];
//    if(num < 2)
//        hiveSignalEdt(guid, num, slot, DB_MODE_SINGLE_VALUE);
//    else
//    {
//        hiveGuid_t joinGuid = hiveEdtCreate(fibJoin, 0, paramc-1, paramv, 2);
//        
//        // Test
//        // hive_tMT_MTHandoverBT(hiveThreadInfo.mastershare_info);
//        // define futures and spawn promises
//        // join at end of EDT
//
//
//
//        u64 args[3] = {joinGuid, 0, num-1};
//        hiveEdtCreate(fibFork, 0, 3, args, 0);
//        
//        args[1] = 1;
//        args[2] = num-2;
//        hiveEdtCreate(fibFork, 0, 3, args, 0);
//    }
//}
//
//hiveGuid_t fibDone(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
//{
//    PRINTF("Fib %u: %u %u\n", paramv[0], depv[0].guid, depc);
//    hiveShutdown();
//}
//
//void initPerNode(unsigned int nodeId, int argc, char** argv)
//{
//    
//}
//
// u64 root_FutureFib[2];
//
//void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
//{   
//    if(!nodeId && !workerId)
//    {
//        unsigned int num = atoi(argv[1]);
//        hiveGuid_t doneGuid = hiveEdtCreate(fibDone, 0, 1, (u64*)&num, 1);
///*
//        { // Iterative version
//        u64 args[3] = {doneGuid, 0, num};   // REQUIRED arg list for iterative solver
//        // hiveGuid_t guid = hiveEdtCreate(fibFork, 0, 3, args, 0); // OPTIONAL non-future iterative
//        hiveGuid_t guid = hiveEdtCreate(fibIterativeFut, 0, 3, args, 0); // OPTIONAL future iterative
//        }
//*/
//        { // Recursive version
//        	u64  args[] = {(u64) NULL, (u64) root_FutureFib, 2, num, num, 0}; // REQUIRED arg list for solver second argument: # futures
//        	hiveGuid_t guid = hiveEdtCreate(fibForkFut, 0, 6, args, 0);
//        }
//    }
//}
//
int main(int argc, char** argv)
{
//    hiveRT(argc, argv);
    return 0;
}
