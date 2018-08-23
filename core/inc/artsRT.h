#ifndef ARTSRT_H
#define ARTSRT_H
#ifdef __cplusplus
extern "C" {
#endif
    
//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
    
/* boolean support in C */
#ifdef __cplusplus
#define TRUE true
#define FALSE false
#else
#define true 1
#define TRUE 1
#define false 0
#define FALSE 0
typedef uint8_t bool;
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

typedef struct
{
    artsGuid_t guid;
    artsType_t mode;
    void *ptr;
} artsEdtDep_t;

typedef void (*artsEdt_t)      (uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]);
typedef void (*eventCallback_t)(artsEdtDep_t data);
typedef void (*sendHandler_t)  (void * args);

typedef enum
{
    ARTS_EVENT_LATCH_DECR_SLOT = 0,
    ARTS_EVENT_LATCH_INCR_SLOT = 1
} artsLatchEventSlot_t;

struct artsHeader
{
    uint8_t type:8;
    uint64_t size:56;
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
    uint32_t paramc;
    uint32_t depc;
    artsGuid_t currentEdt;
    artsGuid_t outputBuffer;
    artsGuid_t epochGuid;
    unsigned int cluster;
    volatile unsigned int depcNeeded;
    volatile unsigned int invalidateCount;
} __attribute__ ((aligned));

struct artsDependent
{
    uint8_t type;
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

typedef struct
{
    unsigned int size;
    unsigned int index;
    artsGuid_t startGuid;
} artsGuidRange;

typedef struct  artsArrayDb
{
    unsigned int elementSize;
    unsigned int elementsPerBlock;
    unsigned int numBlocks;
    char head[];
} artsArrayDb_t;

typedef enum
{
    PHASE_1,
    PHASE_2,
    PHASE_3
} TerminationDetectionPhase;

typedef struct {
    TerminationDetectionPhase phase;
    volatile unsigned int activeCount;
    volatile unsigned int finishedCount;
    volatile unsigned int globalActiveCount;
    volatile unsigned int globalFinishedCount;
    volatile unsigned int lastActiveCount;
    volatile unsigned int lastFinishedCount;
    volatile uint64_t queued;
    volatile uint64_t outstanding;
    unsigned int terminationExitSlot;
    artsGuid_t terminationExitGuid;
    artsGuid_t guid;
    artsGuid_t poolGuid;
    volatile unsigned int * waitPtr;
} artsEpoch_t;

void PRINTF( const char* format, ... );

#ifdef __cplusplus
}
#endif

#endif /* ARTSRT_H */

