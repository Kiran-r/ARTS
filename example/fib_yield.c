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
#include "hive_tMT.h"
uint64_t start = 0;

void fib(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t resultGuid = paramv[0];
    int num = paramv[1];
    int sum = num;
    int x = -1;
    int y = -1;
    
    if(num >= 2) 
    {
        bool hasCtx = 1; //availContext();
        unsigned int ctx = getCurrentContext();
        unsigned int count = 2;
        
//        PRINTF("FIB: %d ctx: %u\n", num, ctx);
        
        int * xPtr = &x;
        int * yPtr = &y;
        artsGuid_t xGuid = artsAllocateLocalBuffer((void**)&xPtr, sizeof(int), 1, NULL_GUID);
        artsGuid_t yGuid = artsAllocateLocalBuffer((void**)&yPtr, sizeof(int), 1, NULL_GUID);
        
        uint64_t args[3];
        
        args[0] = xGuid;
        args[1] = num-2;
        args[2] = ctx;
        artsEdtCreate(fib, 0, (hasCtx) ? 3 : 2, args, 0);
        
        args[0] = yGuid;
        args[1] = num-1;
        artsEdtCreate(fib, 0, (hasCtx) ? 3 : 2, args, 0);
        
        if(hasCtx) 
        {
            artsContextSwitch(2);
        }
        else
        {
            PRINTF("Yield %d\n", num);
            while(x<0 || y<0)
                artsYield();
        }
        sum = x + y;
    }
    
    if(resultGuid)
    {
        artsSetBuffer(resultGuid, &sum, sizeof(int));
        if(paramc == 3)
        {
            setContextAvail(paramv[2]);
        }
    }
    else
    {
        uint64_t time = artsGetTimeStamp() - start;
        PRINTF("Fib %d: %d %lu\n", num, sum, time);
        artsShutdown();
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        int num = atoi(argv[1]);
        uint64_t args[3] = {NULL_GUID, num};
        start = artsGetTimeStamp();
        artsGuid_t guid = artsEdtCreate(fib, 0, 2, args, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}