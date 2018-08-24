//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <sys/resource.h>
#if !defined(__APPLE__)
  #include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <unistd.h>
#include "arts.h"
#include "artsAtomics.h"
#include "artsOutOfOrderList.h"
#include "artsDebug.h"

#define NUMTHREADS 2

typedef void (* callback )( void *, void *);

volatile unsigned int count = 0;
struct artsOutOfOrderList list;

void printer(void * id, void * arg)
{
    unsigned int * Id = (unsigned int *)id;
    printf("Print %u\n", *Id);
    artsFree(Id);
}

void * adder(void * data)
{
    for(unsigned int i=0; i<20; i++)
    {
        unsigned int * id = (unsigned int*) artsMalloc(sizeof(unsigned int));
        *id = i;
        while(!artsOutOfOrderListAddItem(&list, id))
        {
            PRINTF("RESET\n");
            artsOutOfOrderListReset(&list);
        }
//        printf("Print Added %u\n", i);
        artsAtomicAdd(&count, 1U);
    }
}

void * firer(void * data)
{
    while(count < 10);
    PRINTF("FIRE 1\n");
    artsOutOfOrderListFireCallback(&list, 0,  printer);
    while(count < 20);
    PRINTF("FIRE 2\n");
    artsOutOfOrderListFireCallback(&list, 0,  printer);
}

//pthread_create(&nodeThreadList[i], &attr, &artsThreadLoop, &mask[i]);
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
