#include <stdlib.h>
#include "hive.h"
#include "hiveDebug.h"
#include "hiveGlobals.h"
#include "hiveCounter.h"

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
    //PRINTF("Size %d\n", size);
    void * address;
    posix_memalign(&address, 8, size);
    //posix_memalign(&address, 128, size);
    if(address == NULL)
    {
        PRINTF("Out of Memory\n");
        hiveDebugGenerateSegFault();
    }
    zeroMemory(address,size);
    HIVEEDTCOUNTERTIMERENDINCREMENT(mallocMemory);
    return address;
}

extern inline void *
hiveRealloc(void * ptr, size_t size)
{
    return realloc(ptr, size);
}

extern inline void *
hiveCalloc(size_t size)
{
    HIVEEDTCOUNTERTIMERSTART(callocMemory);
    //PRINTF("Size %d\n", size);
    void * address;
    posix_memalign(&address, 8, size);
    //posix_memalign(&address, 128, size);
    zeroMemory(address,size);
    HIVEEDTCOUNTERTIMERENDINCREMENT(callocMemory);
    return address;
}

extern inline void
hiveFree(void *ptr)
{
    HIVEEDTCOUNTERTIMERSTART(freeMemory);
    free(ptr);
    HIVEEDTCOUNTERTIMERENDINCREMENT(freeMemory);
}
