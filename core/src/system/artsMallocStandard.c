#include <stdlib.h>
#include "arts.h"
#include "artsDebug.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsIntrospection.h"

static void zeroMemory(char * addr, size_t size )
{
    size_t i;

    for( i=0; i< size; i++ )
        addr[i]=0;

}

extern inline void *
artsMalloc(size_t size)
{
    ARTSEDTCOUNTERTIMERSTART(mallocMemory);
    size+=sizeof(uint64_t);
    void * address;
    posix_memalign(&address, 8, size);
    if(address == NULL)
    {
        PRINTF("Out of Memory\n");
        artsDebugGenerateSegFault();
    }
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsMallocBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(mallocMemory);
    return address;
}

extern inline void *
artsRealloc(void * ptr, size_t size)
{
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    void * addr = realloc(temp, size + sizeof(uint64_t));
    temp = (uint64_t *) addr;
    *temp = size + sizeof(uint64_t);
    return ++temp;
}

extern inline void *
artsCalloc(size_t size)
{
    ARTSEDTCOUNTERTIMERSTART(callocMemory);
    size+=sizeof(uint64_t);
    void * address;
    posix_memalign(&address, 8, size);
    if(address == NULL)
    {
        PRINTF("Out of Memory\n");
        artsDebugGenerateSegFault();
    }
    zeroMemory(address,size);
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsMallocBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(callocMemory);
    return address;
}

extern inline void
artsFree(void *ptr)
{
    ARTSEDTCOUNTERTIMERSTART(freeMemory);
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    uint64_t size = (*temp);
    free(temp);
    if(artsThreadInfo.mallocTrace)
        artsUpdatePerformanceMetric(artsFreeBW, artsThread, size, false);
    ARTSEDTCOUNTERTIMERENDINCREMENT(freeMemory);
}
