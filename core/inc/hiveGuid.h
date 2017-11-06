#ifndef HIVEGUID_H
#define HIVEGUID_H
#include "hive.h"
#include <stdint.h>

//#define GUID_MAX_KEYS 17179869184
#define GUID_MAX_KEYS 8589934592

typedef union
{
    struct
    {
        u8 isLocal:  1;
        u32 rank:   20; //1 million nodes
        u8  type:    2;
        u64 addr:   41;
    } local;
    struct
    {
        u8 isLocal:  1;
        u32 rank:   15; //1 million nodes
        u8  type:    2;
        u16 thread: 13; //256 threads
        u64 key:    33; //unique key
    } fields;
    intptr_t bits: 64;
} hiveGuid;

void * hiveGuidGetFastPathLocal( hiveGuid_t guid );
hiveGuid_t hiveGuidCreate( void * address );
hiveGuid_t hiveGuidCreateForRank( unsigned int route, unsigned int type );
void hiveGuidTableRemove();
void hiveGuidTableInit( unsigned int routeInitSize );
u32 hiveGuidGetType( hiveGuid_t guid );
unsigned int hiveGuidGetRank( hiveGuid_t guid );
void setGuidGeneratorAfterParallelStart();
void hiveGuidKeyGeneratorInit();
hiveGuid_t hiveGuidCreateForRankInternal( unsigned int route, unsigned int type, unsigned int guidCount);
hiveGuid_t hiveReserveGuidRoute(unsigned int type, unsigned int route);
bool hiveIsGuidLocal(hiveGuid_t guid);
#endif
