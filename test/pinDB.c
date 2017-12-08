#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

unsigned int node = 0;
hiveGuid_t someDbGuid = NULL_GUID;

//This will hang but print a warning if the edt is not on the same node as the pinned DBs
hiveGuid_t edtFunc(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int * ptr = depv[0].ptr;
    unsigned int * ptr2 = depv[1].ptr;
    
    if(*ptr == 1234)
        PRINTF("hiveDbCreate Check\n");
    else
        PRINTF("hiveDBCreate Fail\n");
    
    if(*ptr2 == 9876)
        PRINTF("hiveDbCreateWithGuid Check\n");
    else
        PRINTF("hiveDbCreateWithGuid Fail\n");
    
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    //This is the node we are going to pin to
    node = atoi(argv[1]);
    //Allocate some DB to test hiveDbCreateWithGuid
    someDbGuid = hiveReserveGuidRoute(HIVE_DB, node);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && nodeId == node)
    {
        int * ptr = NULL;
        //Set pin to true to pin to node given by command line
        //It is pinned to the node creating the DB
        hiveGuid_t dbGuid = hiveDbCreate((void**)&ptr, sizeof(unsigned int), true);
        *ptr = 1234;
        
        //EDT is going to run on node given by command line
        hiveGuid_t edtGuid = hiveEdtCreate(edtFunc, node, 0, NULL, 2);
        
        //Put both signals up front forcing one to be out of order to test the OO code path
        hiveSignalEdt(edtGuid, dbGuid, 0, DB_MODE_PIN); //Note the mode
        hiveSignalEdt(edtGuid, someDbGuid, 1, DB_MODE_PIN); //Note the mode
        
        //This is the delayed DB 
        int * ptr2 = hiveDbCreateWithGuid(someDbGuid, sizeof(unsigned int), true);
        *ptr2 = 9876;
    }
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}