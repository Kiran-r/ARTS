#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>
#include "hiveAtomics.h"
#include "hiveMalloc.h"
#include "hiveOutOfOrderList.h"
#include "hiveDebug.h"

#define NUMTHREADS 2

typedef void (* callback )( void *, void *);

volatile unsigned int count = 0;
struct hiveOutOfOrderList list;

void printer(void * id, void * arg)
{
    unsigned int * Id = (unsigned int *)id;
    printf("Print %u\n", *Id);
    hiveFree(Id);
}

void * adder(void * data)
{
    for(unsigned int i=0; i<20; i++)
    {
        unsigned int * id = (unsigned int*) hiveMalloc(sizeof(unsigned int));
        *id = i;
        while(!hiveOutOfOrderListAddItem(&list, id))
        {
            PRINTF("RESET\n");
            hiveOutOfOrderListReset(&list);
        }
//        printf("Print Added %u\n", i);
        hiveAtomicAdd(&count, 1U);
    }
}

void * firer(void * data)
{
    while(count < 10);
    PRINTF("FIRE 1\n");
    hiveOutOfOrderListFireCallback(&list, 0,  printer);
    while(count < 20);
    PRINTF("FIRE 2\n");
    hiveOutOfOrderListFireCallback(&list, 0,  printer);
}

//pthread_create(&nodeThreadList[i], &attr, &hiveThreadLoop, &mask[i]);
//pthread_join(nodeThreadList[i], NULL);

int main(void)
{
    list.head.next = 0;
    for(unsigned int i=0; i<OOPERELEMENT; i++)
    {
        list.head.array[i] = 0;
    }
    
    pthread_t thread[NUMTHREADS];
    for(unsigned int t=0; t<NUMTHREADS-1; t++)
    {
       if(pthread_create(&thread[t], NULL, adder, 0))
       {
          printf("ERROR on create\n");
          exit(-1);
       }
    }
    
    if(pthread_create(&thread[NUMTHREADS-1], NULL, firer, 0))
    {
       printf("ERROR on create\n");
       exit(-1);
    }
    
    void * status;
    for(unsigned int t=0; t<NUMTHREADS; t++) 
    {
        if(pthread_join(thread[t], &status))
        {
           printf("ERROR on join\n");
           exit(-1);
        }
    }
    return 0;
}


