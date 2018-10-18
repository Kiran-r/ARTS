/*
 * hive_tMT_impl.h
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

#ifndef CORE_INC_HIVE_TMT_IMPL_H_
#define CORE_INC_HIVE_TMT_IMPL_H_

#include "hive_tMT.h"



#define DPRINTF( ... )

// types

typedef uint32_t threadid_t;
typedef uint32_t threadunit_t;
typedef uint32_t numtickets_t;

typedef struct
{
  uint32_t threadpool_id;     // alias id
  ti_t*    threadpool_info;   // alias shared pool info
} tmask_t; // per alias thread info

// per alias thread names
typedef void* (*work_to_do_t)();


#endif /* CORE_INC_HIVE_TMT_IMPL_H_ */
