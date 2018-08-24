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
#include <string.h>
#include "arts.h"

unsigned int elemsPerNode = 4;
artsArrayDb_t * array = NULL;

void shutdown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Depc: %u\n", depc);
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elemsPerNode; j++)
        {
            PRINTF("%u: %u\n", i*elemsPerNode+j, data[j]);
        }
    }
    artsShutdown();
}

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGatherArrayDb(array, shutdown, 0, 0, NULL, 0);
}

void edtFunc(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    artsGuid_t checkGuid = paramv[1];
    unsigned int * value = depv[0].ptr;
    *value = index;
    PRINTF("%u:  %u %p\n", index, *value, value);
    artsSignalEdtValue(checkGuid, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        artsGuid_t checkGuid = artsEdtCreate(check, 0, 0, NULL, elemsPerNode * artsGetTotalNodes());
        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elemsPerNode * artsGetTotalNodes());
        artsForEachInArrayDbAtData(array, 1, edtFunc, 1, &checkGuid);
//        artsForEachInArrayDb(array, edtFunc, 1, &checkGuid);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
