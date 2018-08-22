#ifndef ARTSGUID_H
#define ARTSGUID_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include <stdint.h>

//#define GUID_MAX_KEYS 17179869184
#define GUID_MAX_KEYS 8589934592

typedef union
{
    intptr_t bits: 64;
    struct __attribute__((packed))
    {
        u8  type:    8;
        u16 rank:   16; 
        u64 key:    40;
    } fields;
} artsGuid;

//External API
artsGuid_t artsReserveGuidRoute(artsType_t type, unsigned int route);
artsGuid_t artsReserveGuidRouteRemote(artsType_t type, unsigned int route);
bool artsIsGuidLocal(artsGuid_t guid);
unsigned int artsGuidGetRank(artsGuid_t guid);
artsType_t artsGuidGetType(artsGuid_t guid);
artsGuid_t artsGuidCast(artsGuid_t guid, artsType_t type);

//Internal API
artsGuid_t artsGuidCreateForRank(unsigned int route, unsigned int type);
void artsGuidKeyGeneratorInit();
void setGlobalGuidOn();
void setGuidGeneratorAfterParallelStart();

typedef struct
{
    unsigned int size;
    unsigned int index;
    artsGuid_t startGuid;
} artsGuidRange;

artsGuidRange * artsNewGuidRangeNode(unsigned int type, unsigned int size, unsigned int route);
artsGuid_t artsGetGuid(artsGuidRange * range, unsigned int index);
artsGuid_t artsGuidRangeNext(artsGuidRange * range);
bool artsGuidRangeHasNext(artsGuidRange * range);
void artsGuidRangeResetIter(artsGuidRange * range);

#ifdef __cplusplus
}
#endif

#endif
