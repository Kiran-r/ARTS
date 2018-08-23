#ifndef ARTSGUID_H
#define ARTSGUID_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"

typedef union
{
    intptr_t bits: 64;
    struct __attribute__((packed))
    {
        uint8_t  type:    8;
        uint16_t rank:   16; 
        uint64_t key:    40;
    } fields;
} artsGuid;

artsGuid_t artsGuidCreateForRank(unsigned int route, unsigned int type);
void artsGuidKeyGeneratorInit();
void setGlobalGuidOn();
void setGuidGeneratorAfterParallelStart();

#ifdef __cplusplus
}
#endif

#endif
