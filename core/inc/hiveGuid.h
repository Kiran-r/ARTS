#ifndef HIVEGUID_H
#define HIVEGUID_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hive.h"
#include <stdint.h>

//#define GUID_MAX_KEYS 17179869184
#define GUID_MAX_KEYS 8589934592

typedef union
{
    intptr_t bits: 64;
    struct __attribute__((packed))
    {
        u8  local:   1; //Means it is created locally
        u8  type:    5;
        u16 rank:   13; //1 million nodes
        u16 thread: 13; //256 threads
        u32 key:    32; //unique key
    } fields;
} hiveGuid;

//External API
hiveGuid_t hiveReserveGuidRoute(hiveType_t type, unsigned int route);
hiveGuid_t hiveReserveGuidRouteRemote(hiveType_t type, unsigned int route);
bool hiveIsGuidLocal(hiveGuid_t guid);
unsigned int hiveGuidGetRank(hiveGuid_t guid);
hiveType_t hiveGuidGetType(hiveGuid_t guid);
hiveType_t hiveGuidCast(hiveGuid_t guid, hiveType_t type);

//Internal API
hiveGuid_t hiveGuidCreateForRank(unsigned int route, unsigned int type);
void hiveGuidKeyGeneratorInit();
void setGuidGeneratorAfterParallelStart();

typedef struct
{
    unsigned int size;
    unsigned int index;
    hiveGuid_t startGuid;
} hiveGuidRange;

hiveGuidRange * hiveNewGuidRangeNode(unsigned int type, unsigned int size, unsigned int route);
hiveGuid_t hiveGetGuid(hiveGuidRange * range, unsigned int index);
hiveGuid_t hiveGuidRangeNext(hiveGuidRange * range);
bool hiveGuidRangeHasNext(hiveGuidRange * range);
void hiveGuidRangeResetIter(hiveGuidRange * range);

#ifdef __cplusplus
}
#endif

#endif
