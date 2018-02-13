/*
 * This file is subject to the license agreement located in the file LICENSE
 * and cannot be distributed without it. This notice cannot be
 * removed or modified.
 */

#ifndef HIVE_H
#define HIVE_H
#ifdef __cplusplus
extern "C" {
#endif

#define _GNU_SOURCE

#include "stdint.h"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

void PRINTF( const char* format, ... );

#define HIVE_EDT 0
#define HIVE_EVENT 1
#define HIVE_DB 2
#define HIVE_CALLBACK 3

typedef uint64_t u64; /**< 64-bit unsigned integer */
typedef uint32_t u32; /**< 32-bit unsigned integer */
typedef uint16_t u16; /**< 16-bit unsigned integer */
typedef uint8_t u8;   /**< 8-bit unsigned integer */
typedef int64_t s64;  /**< 64-bit signed integer */
typedef int32_t s32;  /**< 32-bit signed integer */
typedef int8_t s8;    /**< 8-bit signed integer */

/* boolean support in C */
#ifdef __cplusplus
#define TRUE true
#define FALSE false
#else
#define true 1
#define TRUE 1
#define false 0
#define FALSE 0
typedef u8 bool;
#endif /* __cplusplus */


typedef intptr_t hiveGuid_t; /**< GUID type */
#define NULL_GUID ((hiveGuid_t)0x0)

typedef enum
{
    DB_MODE_NON_COHERENT_READ = 0,
    DB_MODE_NON_COHERENT_WRITE,
    DB_MODE_CDAG_WRITE,
    DB_MODE_EXCLUSIVE_READ,
    DB_MODE_EXCLUSIVE_WRITE,
    DB_MODE_SINGLE_VALUE,
    DB_MODE_PIN,
    DB_MODE_PTR
} hiveDbAccessMode_t;

typedef struct
{
    hiveGuid_t guid;
    hiveDbAccessMode_t mode;
    void *ptr;
} hiveEdtDep_t;

typedef hiveGuid_t(*hiveEdt_t) (u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

typedef void (*sendHandler_t) (void * args);

typedef enum
{
    HIVE_EVENT_ONCE_T,/**< A ONCE event simply passes along a satisfaction on its
                          * unique pre-slot to its post-slot. Once all hive objects
                          * linked to its post-slot have been satisfied, the ONCE event
                          * is automatically destroyed. */
    HIVE_EVENT_IDEM_T,/**< An IDEM event simply passes along a satisfaction on its
                          * unique pre-slot to its post-slot. The IDEM event persists
                          * until hiveEventDestroy() is explicitly called on it.
                          * It can only be satisfied once and susequent
                          * satisfactions are ignored (use case: BFS, B&B..) */
    HIVE_EVENT_STICKY_T,
    /**< A STICKY event is identical to an IDEM event except that
                 * multiple satisfactions result in an error
                 */
    HIVE_EVENT_LATCH_T,
    /**< A LATCH event has two pre-slots: a INCR and a DECR.
                 * Each slot is associated with an internal monotonically
                 * increasing counter that starts at 0. On each satisfaction
                 * of one of the pre-slots, the counter for that slot is
                 * incremented by 1. When both counters are equal (and non-zero),
                 * the post-slot of the latch event is triggered.
                 * Any data block passed along its pre-slots is ignored.
                 * A LATCH event has the same persistent as a ONCE event and
                 * is automatically destroyed when its post-slot is triggered.
                 */
    HIVE_EVENT_LOOP_T,
    
    HIVE_EVENT_PIPELINE_T,

    HIVE_EVENT_T_MAX	 /**< This is *NOT* an event and is only used to count
                          * the number of event types. Its use is reserved for the
                          * runtime. */
} hiveEventTypes_t;

typedef enum
{
    HIVE_EVENT_LATCH_DECR_SLOT = 0,
    HIVE_EVENT_LATCH_INCR_SLOT = 1
} hiveLatchEventSlot_t;

typedef void (*eventCallback_t)(hiveEdtDep_t data);

struct hiveHeader
{
    u8 type:8;
    u64 size:56;
} __attribute__ ((aligned));

struct hiveDb
{
    struct hiveHeader header;
    hiveGuid_t guid;
    void * dbList;
} __attribute__ ((aligned));

struct hiveEdt
{
    struct hiveHeader header;
    hiveEdt_t funcPtr;
    u32 paramc;
    u32 depc;
    hiveGuid_t currentEdt;
    hiveGuid_t outputEvent;
    hiveGuid_t epochGuid;
    volatile unsigned int depcNeeded;
    volatile unsigned int invalidateCount;
} __attribute__ ((aligned));

struct hiveDependent
{
    u8 type;
    //unsigned int * volatile counter;
    volatile unsigned int slot;
    volatile hiveGuid_t addr;
    volatile eventCallback_t callback;
    volatile bool doneWriting;
    //volatile unsigned int lock;
    hiveDbAccessMode_t mode;
};

struct hiveDependentList
{
    unsigned int size;
    //volatile unsigned int resize;
    struct hiveDependentList * volatile next;
    struct hiveDependent dependents[];
};

struct hiveDelayedSatisfyItem
{
    hiveGuid_t src;
    hiveGuid_t dest;
    volatile bool doneWriting;
};

struct hiveDelayedSatisfyList
{
    unsigned int size;
    struct hiveDelayedSatisfyList * volatile next;
    struct hiveDelayedSatisfyItem item[]; 
};

struct hiveDelayedSatisfy
{
    volatile bool fired;
    volatile unsigned int canSatisfy; 
    volatile unsigned int reuse; 
    volatile unsigned int firedCount;
    volatile unsigned int lastKnownCount;
    volatile unsigned int satisfyCount;
    struct hiveDelayedSatisfyList list;
};

struct hiveEvent
{
    struct hiveHeader header;
    hiveEventTypes_t eventType;
    volatile bool fired;
    volatile unsigned int destroyOnFire;
    volatile unsigned int latchCount;
    volatile unsigned int pos;
    volatile unsigned int lastKnown;
    volatile unsigned int dependentCount;
    hiveGuid_t data;
    struct hiveDependentList dependent;
} __attribute__ ((aligned));

void hiveShutdown();

#ifdef __cplusplus
}
#endif
#endif
