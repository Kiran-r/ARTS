#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsRemoteFunctions.h"

unsigned int numElements = 0;

void sendHandler(void * args)
{
    bool pass = true;
    unsigned int * data = args;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(data[i]!=i)
            pass = false;
    }
    if(pass)
        PRINTF("CHECK %u of %u\n", artsGetCurrentNode(), artsGetTotalNodes());
    
    if(artsGetCurrentNode() + 1 == artsGetTotalNodes())
    {
        PRINTF("Shutdown\n");
        artsShutdown();
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numElements = atoi(argv[1]);
    if(!nodeId)
    {
        unsigned int size = sizeof(unsigned int)*numElements;
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            unsigned int * data = artsMalloc(size);
            for(unsigned int j=0; j<numElements; j++)
                data[j] = j;
            artsRemoteSend(i, sendHandler, data, size, true);
        }
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}