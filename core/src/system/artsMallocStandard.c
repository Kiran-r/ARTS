#include "artsRT.h"
#include "artsDebug.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsIntrospection.h"

static void zeroMemory(char * addr, size_t size )
{
    for(size_t i=0; i< size; i++ )
        addr[i]=0;
}

void * artsMalloc(size_t size)
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

void * artsRealloc(void * ptr, size_t size)
{
    uint64_t * temp = (uint64_t*) ptr;
    temp--;
    void * addr = realloc(temp, size + sizeof(uint64_t));
    temp = (uint64_t *) addr;
    *temp = size + sizeof(uint64_t);
    return ++temp;
}

void * artsCalloc(size_t size)
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

void artsFree(void *ptr)
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

void * artsMallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = artsMalloc(size + align);
    memset(ptr, 0, align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 'a';
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void * artsCallocAlign(size_t size, size_t align)
{
    if(!size || align < ALIGNMENT || align % 2)
        return NULL;

    void * ptr = artsCalloc(size + align);
    if(ptr)
    {
        char * temp = ptr;
        *temp = 1;
        ptr = (void*)(temp+1);
        uintptr_t mask = ~(uintptr_t)(align - 1);
        ptr = (void *)(((uintptr_t)ptr + align - 1) & mask);
    }
    return ptr;
}

void artsFreeAlign(void * ptr)
{
    char * trail = (char*)ptr - 1;
    while(!(*trail))
        trail--;
    artsFree(trail);
}