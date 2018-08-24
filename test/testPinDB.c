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

unsigned int node = 0;
artsGuid_t someDbGuid = NULL_GUID;

//This will hang but print a warning if the edt is not on the same node as the pinned DBs
void edtFunc(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * ptr = depv[0].ptr;
    unsigned int * ptr2 = depv[1].ptr;
    
    if(*ptr == 1234)
        PRINTF("artsDbCreate Check\n");
    else
        PRINTF("artsDBCreate Fail\n");
    
    if(*ptr2 == 9876)
        PRINTF("artsDbCreateWithGuid Check\n");
    else
        PRINTF("artsDbCreateWithGuid Fail\n");
    
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    //This is the node we are going to pin to
    node = atoi(argv[1]);
    //Allocate some DB to test artsDbCreateWithGuid
    someDbGuid = artsReserveGuidRoute(ARTS_DB_PIN, node);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && nodeId == node)
    {
        int * ptr = NULL;
        //Set pin to true to pin to node given by command line
        //It is pinned to the node creating the DB
        artsGuid_t dbGuid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_PIN);
        *ptr = 1234;
        
        //EDT is going to run on node given by command line
        artsGuid_t edtGuid = artsEdtCreate(edtFunc, node, 0, NULL, 2);
        
        //Put both signals up front forcing one to be out of order to test the OO code path
        artsSignalEdt(edtGuid, 0, dbGuid); //Note the mode
        artsSignalEdt(edtGuid, 1, someDbGuid); //Note the mode
        
        //This is the delayed DB 
        int * ptr2 = artsDbCreateWithGuid(someDbGuid, sizeof(unsigned int));
        *ptr2 = 9876;
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}