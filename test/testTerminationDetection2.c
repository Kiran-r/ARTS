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

artsArrayDb_t * array = NULL;
artsGuid_t arrayGuid = NULL_GUID;
unsigned int elements = 32;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        PRINTF("%u: %u\n", i, *data);
    }
    
    artsShutdown();
}

void gatherTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Gather task\n");
    artsGuid_t edtGuid = artsEdtCreate(check, 0, 0, NULL, elements);
    for(unsigned int i=0; i<elements; i++)
        artsGetFromArrayDb(edtGuid, i, array, i);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    elements = (unsigned int) atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId) {
        artsGuid_t gatherGuid = artsEdtCreate(gatherTask, 0, 0, 0, 1);
        artsInitializeAndStartEpoch(gatherGuid, 0);
        arrayGuid = artsNewArrayDb(&array, sizeof(unsigned int), elements);
        for(unsigned int i=0; i<elements; i++)
        {
            artsPutInArrayDb(&i, NULL_GUID, 0, array, i);
        }
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
