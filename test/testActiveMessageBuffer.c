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

artsGuid_t dbDestGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

void setter(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    
    unsigned int id = paramv[0];
    unsigned int * buffer = depv[0].ptr;
    unsigned int * dest = depv[1].ptr;
    for(unsigned int i=0; i<blockSize; i++)
    {
        dest[id*blockSize + i] = buffer[i];
    }
    PRINTF("Setter: %u\n", id);
    artsSignalEdt(shutdownGuid, id, dbDestGuid);
}

void getter(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int id = paramv[0];
    unsigned int * source = depv[0].ptr;
    unsigned int * buffer = &source[id*blockSize];
    PRINTF("Getter: %u\n", id);
    //This one actually sends to a remote node... yea for testing!
    artsGuid_t am = artsActiveMessageWithBuffer(setter, artsGetTotalNodes() - 1, paramc, paramv, 1, buffer, sizeof(unsigned int)*blockSize);
    artsSignalEdt(am, 1, dbDestGuid);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    bool pass = true;
    unsigned int * data = depv[0].ptr;
    for(unsigned int i=0; i<numElements; i++)
    {
        if(data[i]!=i)
        {
            PRINTF("I: %u vs %u\n", i, data[i]);
            pass = false;
        }
    }
    
    if(pass)
        PRINTF("CHECK\n");
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    blockSize = atoi(argv[1]);
    numElements = blockSize * artsGetTotalNodes();
    dbDestGuid = artsReserveGuidRoute(ARTS_DB_PIN, artsGetTotalNodes() - 1);
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, artsGetTotalNodes() - 1);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {   
        uint64_t id = nodeId;
        unsigned int * data = artsMalloc(sizeof(unsigned int)*numElements);
        for(unsigned int i=0; i<numElements; i++)
        {
            data[i] = i;
        }
        //This is kinda dumb since it is sending to itself, but hey lets check it...
        artsActiveMessageWithBuffer(getter, nodeId, 1, &id, 0, data, sizeof(unsigned int)*numElements);
        
        if(!nodeId)
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes());
        
        if(nodeId == artsGetTotalNodes() - 1)
            artsDbCreateWithGuid(dbDestGuid, sizeof(unsigned int) * numElements);
    }
}


int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}