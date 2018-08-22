/*
 * This file is subject to the license agreement located in the file LICENSE
 * and cannot be distributed without it. This notice cannot be
 * removed or modified.
 */

#ifndef ARTS_H
#define ARTS_H
#ifdef __cplusplus
extern "C" {
#endif

  // #define _GNU_SOURCE

#include "stdint.h"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

void PRINTF( const char* format, ... );

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

typedef intptr_t artsGuid_t; /**< GUID type */
#define NULL_GUID ((artsGuid_t)0x0)

typedef enum
{
    ARTS_NULL = 0,
    ARTS_EDT,
    ARTS_EVENT,
    ARTS_EPOCH,
    ARTS_CALLBACK,
    ARTS_BUFFER,
    ARTS_DB_READ,
    ARTS_DB_WRITE,
    ARTS_DB_PIN,
    ARTS_DB_ONCE,
    ARTS_DB_ONCE_LOCAL,
    ARTS_LAST_TYPE,
    ARTS_SINGLE_VALUE,
    ARTS_PTR
} artsType_t;

#define artsTypeName const char * const _artsTypeName[] = { \
"ARTS_NULL", \
"ARTS_EDT", \
"ARTS_EVENT", \
"ARTS_EPOCH", \
"ARTS_CALLBACK", \
"ARTS_BUFFER", \
"ARTS_DB_READ", \
"ARTS_DB_WRITE", \
"ARTS_DB_PIN", \
"ARTS_DB_ONCE", \
"ARTS_DB_ONCE_LOCAL", \
"ARTS_LAST_TYPE", \
"ARTS_SINGLE_VALUE", \
"ARTS_PTR" }

#define getTypeName(x) _artsTypeName[x]

typedef struct
{
    artsGuid_t guid;
    artsType_t mode;
    void *ptr;
} artsEdtDep_t;

typedef artsGuid_t (*artsEdt_t) (u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]);

typedef void (*sendHandler_t) (void * args);

typedef enum
{
    ARTS_EVENT_LATCH_DECR_SLOT = 0,
    ARTS_EVENT_LATCH_INCR_SLOT = 1
} artsLatchEventSlot_t;

typedef void (*eventCallback_t)(artsEdtDep_t data);

struct artsHeader
{
    u8 type:8;
    u64 size:56;
} __attribute__ ((aligned));

struct artsDb
{
    struct artsHeader header;
    artsGuid_t guid;
    void * dbList;
} __attribute__ ((aligned));

struct artsEdt
{
    struct artsHeader header;
    artsEdt_t funcPtr;
    u32 paramc;
    u32 depc;
    artsGuid_t currentEdt;
    artsGuid_t outputEvent;
    artsGuid_t epochGuid;
    unsigned int cluster;
    volatile unsigned int depcNeeded;
    volatile unsigned int invalidateCount;
} __attribute__ ((aligned));

struct artsDependent
{
    u8 type;
    volatile unsigned int slot;
    volatile artsGuid_t addr;
    volatile eventCallback_t callback;
    volatile bool doneWriting;
};

struct artsDependentList
{
    unsigned int size;
    struct artsDependentList * volatile next;
    struct artsDependent dependents[];
};

struct artsEvent
{
    struct artsHeader header;
    volatile bool fired;
    volatile unsigned int destroyOnFire;
    volatile unsigned int latchCount;
    volatile unsigned int pos;
    volatile unsigned int lastKnown;
    volatile unsigned int dependentCount;
    artsGuid_t data;
    struct artsDependentList dependent;
} __attribute__ ((aligned));

void artsShutdown();

#ifdef __cplusplus
}
#endif
#endif
