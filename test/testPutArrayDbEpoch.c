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
artsArrayDb_t * array = NULL;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u\n", data[j]);
        }
    }
    artsShutdown();
}

void epochEnd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{    
    artsGatherArrayDb(array, check, 0, 0, NULL, 0);
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
}

void epochStart(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Launching\n");
    artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elementsPerBlock * blocks);
    for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
    {
        artsPutInArrayDb(&i, NULL_GUID, 0, array, i);
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = artsGetTotalNodes();
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!nodeId && !workerId)
    {
        artsGuid_t endEpochGuid = artsEdtCreate(epochEnd, 0, 0, NULL, 1);
        artsInitializeAndStartEpoch(endEpochGuid, 0);
        artsGuid_t startEpochGuid = artsEdtCreate(epochStart, 0, 0, NULL, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
