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
#include "artsAtomics.h"

unsigned int counter = 0;
unsigned int numDummy = 0;
artsGuid_t exitGuid = NULL_GUID;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    artsAtomicAdd(&counter, 1);
}

void exitProgram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    unsigned int numNodes = artsGetTotalNodes();
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int numEdts = depv[i].guid;
        if(numEdts!=numNodes*numDummy+2)
            PRINTF("Error: %u vs %u\n", numEdts, numNodes*numDummy+2);
    }
    PRINTF("Exit %u\n", counter);
    artsShutdown();
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    artsGuid_t guid = artsGetCurrentEpochGuid();
    PRINTF("Starting %lu %u\n", guid, artsGuidGetRank(guid));
    unsigned int numNodes = artsGetTotalNodes();
    for (unsigned int rank = 0; rank < numNodes * numDummy; rank++)
        artsEdtCreate(dummytask, rank % numNodes, 0, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (unsigned int) atoi(argv[1]);
    exitGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId)
        artsEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, artsGetTotalNodes());
    
    if (!workerId) 
    {
        artsInitializeAndStartEpoch(exitGuid, nodeId);
        artsGuid_t startGuid = artsEdtCreate(rootTask, nodeId, 0, NULL, 0);
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
