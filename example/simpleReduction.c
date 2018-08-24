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

unsigned int numDbs = 0;
artsGuid_t reductionGuid = NULL_GUID;

void reduction(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t total = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        int * dbPtr = depv[i].ptr;
        total+=dbPtr[0];
    }
    artsSignalEdtValue(paramv[0], 0, total);
}

void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Result %lu\n", depv[0].guid);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numDbs = artsGetTotalNodes();  
    reductionGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        int * ptr;
        artsGuid_t dbGuid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_READ);
        *ptr = nodeId;
        
        artsSignalEdt(reductionGuid, nodeId, dbGuid);
        
        if(!nodeId)
        {
            artsGuid_t guid = artsEdtCreate(shutDown, 0, 0, NULL, 1);
            artsEdtCreateWithGuid(reduction, reductionGuid, 1, (uint64_t*)&guid, numDbs);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}