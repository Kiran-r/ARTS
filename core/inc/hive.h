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

  // #define _GNU_SOURCE

#include "stdint.h"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

void PRINTF( const char* format, ... );

#define HIVE_EDT 0
#define HIVE_EPOCH 0
#define HIVE_EVENT 1
#define HIVE_DB 2
#define HIVE_CALLBACK 3
#define HIVE_BUFFER 3

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
    DB_MODE_NONE = -1,
    DB_MODE_NON_COHERENT_READ = 0,
    DB_MODE_NON_COHERENT_WRITE,
    DB_MODE_CDAG_WRITE,
    DB_MODE_EXCLUSIVE_READ,
    DB_MODE_EXCLUSIVE_WRITE,
    DB_MODE_SINGLE_VALUE,
    DB_MODE_PIN,
    DB_MODE_PTR,
    DB_MODE_ONCE_LOCAL,
    DB_MODE_ONCE
} hiveDbAccessMode_t;

typedef struct
{
    hiveGuid_t guid;
    hiveDbAccessMode_t mode;
    void *ptr;
} hiveEdtDep_t;

typedef hiveGuid_t (*hiveEdt_t) (u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

typedef void (*sendHandler_t) (void * args);

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
    hiveDbAccessMode_t mode;
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
    unsigned int cluster;
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

struct hiveEvent
{
    struct hiveHeader header;
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
