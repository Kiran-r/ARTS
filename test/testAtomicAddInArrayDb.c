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

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
unsigned int numAdd = 0;
artsArrayDb_t * array = NULL;
artsGuid_t arrayGuid = NULL_GUID;

void end(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc-1; i++)
    {
        unsigned int data = depv[i].guid;
        PRINTF("updates: %u\n", data);
    }
    artsShutdown();
}

//Created by the epochEnd via gather will signal end
void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u j: %u %u\n", i, j, data[j]);
        }
    }
    artsSignalEdtValue(paramv[0], numAdd*elementsPerBlock*blocks, 0);
}

//This is run at the end of the epoch
void epochEnd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{    
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
    artsGatherArrayDb(array, check, 0, 1, paramv, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = artsGetTotalNodes();
    numAdd = atoi(argv[2]);
    arrayGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && !nodeId)
    {
        //The end will get all the updates and a signal from the gather
        artsGuid_t endGuid = artsEdtCreate(end, 0, 0, NULL, numAdd*elementsPerBlock*blocks + 1);

        artsGuid_t endEpochGuid = artsEdtCreate(epochEnd, 0, 1, &endGuid, 1);
        artsInitializeAndStartEpoch(endEpochGuid, 0);

        array = artsNewArrayDbWithGuid(arrayGuid, sizeof(unsigned int), elementsPerBlock * blocks);

        for(unsigned int j=0; j<numAdd; j++)
        {
            for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            {
                PRINTF("i: %u Slot: %u edt: %lu\n", i, j*elementsPerBlock*blocks + i, endGuid);
                artsAtomicAddInArrayDb(array, i, 1, endGuid, j*elementsPerBlock*blocks + i);
            }
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
