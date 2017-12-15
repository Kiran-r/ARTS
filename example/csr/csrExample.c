#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"
#include "distributedCsr.h"

hiveGuid_t * graphGuids;
hiveGuid_t * printGuids;

hiveGuid_t printer(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    graph_csr_distr * graph = depv[0].ptr;
    printGraph(graph);
    
    if(hiveGetCurrentNode() + 1 == hiveGetTotalNodes())
        hiveShutdown();
    else
        hiveSignalEdt(printGuids[hiveGetCurrentNode()+1], graphGuids[hiveGetCurrentNode()+1], 0, DB_MODE_PIN);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    char * fileName = argv[1];
    u64 scale = atoi(argv[2]);
    u64 numVert = 1 << scale;
    
    graphGuids = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        graphGuids[i] = hiveReserveGuidRoute(HIVE_DB, i);

    PRINTF("Reading...\n");
    u64 nodeVert = readGraph(fileName, graphGuids[nodeId], nodeId, hiveGetTotalNodes(), numVert);
    PRINTF("Scale: %u numVert: %ld nodeVert: %ld\n", scale, numVert, nodeVert);
    
    printGuids = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        printGuids[i] = hiveReserveGuidRoute(HIVE_DB, i);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        hiveEdtCreateWithGuid(printer, printGuids[nodeId], 0, NULL, 1);
        
        if(!nodeId)
        {
            hiveSignalEdt(printGuids[0], graphGuids[0], 0, DB_MODE_PIN);
        }
    }
}


int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}