#include <stdlib.h>
#include "hive.h"
#include "hiveDebug.h"
#include "hiveGlobals.h"
#include "hiveCounter.h"
#include "hiveIntrospection.h"

static void zeroMemory(char * addr, size_t size )
{
    size_t i;

    for( i=0; i< size; i++ )
        addr[i]=0;

}

extern inline void *
hiveMalloc(size_t size)
{
    HIVEEDTCOUNTERTIMERSTART(mallocMemory);
    size+=sizeof(uint64_t);
    void * address;
    posix_memalign(&address, 8, size);
    if(address == NULL)
    {
        PRINTF("Out of Memory\n");
        hiveDebugGenerateSegFault();
    }
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    hiveUpdatePerformanceMetric(hiveMallocBW, hiveThread, size, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(mallocMemory);
    return address;
}

extern inline void *
hiveRealloc(void * ptr, size_t size)
{
    return NULL; //realloc(ptr, size);
}

extern inline void *
hiveCalloc(size_t size)
{
    HIVEEDTCOUNTERTIMERSTART(callocMemory);
    size+=sizeof(uint64_t);
    void * address;
    posix_memalign(&address, 8, size);
    if(address == NULL)
    {
        PRINTF("Out of Memory\n");
        hiveDebugGenerateSegFault();
    }
    zeroMemory(address,size);
    uint64_t * temp = (uint64_t*) address;
    *temp = size;
    address = (void*)(temp+1);
    hiveUpdatePerformanceMetric(hiveMallocBW, hiveThread, size, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(callocMemory);
    return address;
}

extern inline void
hiveFree(void *ptr)
{
    HIVEEDTCOUNTERTIMERSTART(freeMemory);
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    uint64_t size = (*temp);
    free(temp);
    hiveUpdatePerformanceMetric(hiveFreeBW, hiveThread, size, false);
    HIVEEDTCOUNTERTIMERENDINCREMENT(freeMemory);
}
