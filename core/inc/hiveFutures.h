/*
 * hiveFutures.h
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

#ifndef CORE_INC_HIVEFUTURES_H_
#define CORE_INC_HIVEFUTURES_H_

// function declarations
// USER interface
typedef uint64_t ticket_t;

// returns a ticket for the outstanding future; create DB: future storage created by callee
ticket_t hiveCreateFuture(uint32_t paramc, uint64_t* paramv); // first parameter is pointer to future
// retrieves future with ticket
void hiveGetFuture(ticket_t ticket);
void hiveGetFutures(ticket_t* ticket, unsigned int num);

// for debugging purposes
uint32_t ext_threadID();
uint32_t ext_threadUNIT();


#endif /* CORE_INC_HIVEFUTURES_H_ */
