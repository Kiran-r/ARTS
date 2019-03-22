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
uint64_t numDummy = 0;
artsGuid_t exitGuid = NULL_GUID;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
}

void syncTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{ 
    PRINTF("Guid: %lu Sync %lu: %lu\n", artsGetCurrentGuid(), paramv[0], depv[0].guid);
}

void exitProgram(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    PRINTF("Exit: %lu\n", depv[0].guid);
    artsShutdown();
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    uint64_t dep = paramv[0];
    if(dep)
    {
        dep--;
//        artsGuid_t guid = artsEdtCreate(syncTask, artsGetCurrentNode(), 1, &dep, 1);
//        artsGuid_t epochGuid = artsInitializeAndStartEpoch(guid, 0);
        artsGuid_t epochGuid = artsInitializeAndStartEpoch(NULL_GUID, 0);
        PRINTF("Guid: %lu Root: %lu sync: %lu epoch: %lu\n", artsGetCurrentGuid(), dep, NULL_GUID, epochGuid);
        
        unsigned int numNodes = artsGetTotalNodes();
        for (unsigned int rank = 0; rank < numNodes; rank++)
            artsEdtCreate(rootTask, rank % numNodes, 1, &dep, 0);
        
        for (uint64_t rank = 0; rank < numNodes * numDummy; rank++)
            artsEdtCreate(dummytask, rank % numNodes, 0, NULL, 0);
        
        artsWaitOnHandle(epochGuid);
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (uint64_t) atoi(argv[1]);
    exitGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId)
    {
        if(!workerId)
        {
            PRINTF("Starting\n");
            artsEdtCreateWithGuid(exitProgram, exitGuid, 0, NULL, 1);
            artsGuid_t epochGuid = artsInitializeAndStartEpoch(exitGuid, 0);
            artsGuid_t startGuid = artsEdtCreate(rootTask, 0, 1, &numDummy, 0);
        }
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
