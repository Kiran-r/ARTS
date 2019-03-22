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

uint64_t start = 0;

void fibJoin(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int x = depv[0].guid;
    unsigned int y = depv[1].guid;
    artsSignalEdtValue(paramv[0], paramv[1], x+y);
}

void fibFork(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int next = 0; //(artsGetCurrentNode() + 1) % artsGetTotalNodes();
//    PRINTF("NODE: %u WORKER: %u NEXT: %u\n", artsGetCurrentNode(), artsGetCurrentWorker(), next);
    artsGuid_t guid = paramv[0];
    unsigned int slot = paramv[1];
    unsigned int num = paramv[2];
    if(num < 2)
        artsSignalEdtValue(guid, slot, num);
    else
    {
        artsGuid_t joinGuid = artsEdtCreate(fibJoin, 0, paramc-1, paramv, 2);
        
        uint64_t args[3] = {joinGuid, 0, num-1};
        artsEdtCreate(fibFork, next, 3, args, 0);
        
        args[1] = 1;
        args[2] = num-2;
        artsEdtCreate(fibFork, next, 3, args, 0);
    }
}

void fibDone(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t time = artsGetTimeStamp() - start;
    PRINTF("Fib %u: %u %lu\n", paramv[0], depv[0].guid, time);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        unsigned int num = atoi(argv[1]);
        artsGuid_t doneGuid = artsEdtCreate(fibDone, 0, 1, (uint64_t*)&num, 1);
        uint64_t args[3] = {doneGuid, 0, num};
        start = artsGetTimeStamp();
        artsGuid_t guid = artsEdtCreate(fibFork, 0, 3, args, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}