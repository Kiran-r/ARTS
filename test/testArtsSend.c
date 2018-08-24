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