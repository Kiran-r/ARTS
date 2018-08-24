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
artsGuid_t shutdownGuid;
artsGuid_t * guids;

void shutdownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsShutdown();
}

void acquireTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * num = depv[i].ptr;
        printf("%u %u i: %u %u\n", artsGetCurrentNode(), artsGetCurrentWorker(), i, *num);
    }
    artsSignalEdtValue(shutdownGuid, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guids = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        guids[i] = artsReserveGuidRoute(ARTS_DB_READ, i);
        if(!nodeId)
            PRINTF("i: %u guid: %ld\n", i, guids[i]);
    }
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            if(artsIsGuidLocal(guids[i]))
            {
                unsigned int * ptr = artsDbCreateWithGuid(guids[i], sizeof(unsigned int));
                *ptr = i;
                PRINTF("Created i: %u guid: %ld\n", i, guids[i]);
            }
        }
        
        if(!nodeId)
        {
            artsEdtCreateWithGuid(shutdownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes()*artsGetTotalWorkers());     
        }
    }
    artsGuid_t edtGuid = artsEdtCreate(acquireTest, nodeId, 0, NULL, artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        artsSignalEdt(edtGuid, i, guids[i]);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

