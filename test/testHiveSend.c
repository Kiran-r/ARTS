#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

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
        PRINTF("CHECK %u of %u\n", hiveGetCurrentNode(), hiveGetTotalNodes());
    
    if(hiveGetCurrentNode() + 1 == hiveGetTotalNodes())
    {
        PRINTF("Shutdown\n");
        hiveShutdown();
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numElements = atoi(argv[1]);
    if(!nodeId)
    {
        unsigned int size = sizeof(unsigned int)*numElements;
        for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        {
            unsigned int * data = hiveMalloc(size);
            for(unsigned int j=0; j<numElements; j++)
                data[j] = j;
            hiveRemoteSend(i, sendHandler, data, size, true);
        }
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}