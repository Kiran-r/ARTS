#include <stdio.h>
#include <stdlib.h>
#include "hiveRT.h"

#define NUMVERT 100
#define NUMEDGE 10

hiveGuid_t * graphGuid;

typedef struct
{
    unsigned int target;
    unsigned int weight;
} edge;

typedef struct
{
    unsigned int index;
    unsigned int distance;
    edge edgeList[NUMEDGE];
} vertex;

hiveGuid_t indexToGuid(unsigned int index)
{
    return graphGuid[index/NUMVERT];
}

unsigned int globalIndexToOffset(unsigned int index)
{
    return index % NUMVERT;
}

hiveGuid_t visit(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    u64 * neighbors = &paramv[1];
    vertex * home = depv[0].ptr;
    unsigned int * distance = &home[index].distance;
    
    for(unsigned int i=1; i<paramc; i++)
    {
        vertex * current = depv[i].ptr;
        unsigned int offset = globalIndexToOffset(paramc[i]);
        unsigned int temp = current[offset].distance + ;
    }
    
    
}

hiveGuid_t getDistances(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    vertex * v = depv[0].ptr;
    for(unsigned int i=0; i<NUMVERT; i++)
    {
        u64 edges[NUMEDGE+1];
        edges[0] = i;
        for(unsigned int j=0; j<NUMEDGE; j++)
            edges[j+1] = v[i].edgeList[j].target;
        
        hiveGuid_t guid = hiveEdtCreate(visit, 0, NUMEDGE+1, edges, NUMEDGE+1, NULL);
        
        for(unsigned int j=0; j<NUMEDGE; j++)
            hiveSignalEdt(guid, indexToGuid(v[i].edgeList[j].target), j+1, DB_MODE_NON_COHERENT_READ);
        
        hiveSignalEdt(guid, indexToGuid(v[i].index), 0, DB_MODE_NON_COHERENT_READ);
    }
    hiveShutdown();
}

hiveGuid_t shutDown(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        vertex * v = depv[i].ptr;
        for(unsigned int j=0; j<NUMVERT; j++)
        {
            for(unsigned int k=0; k<NUMEDGE; k++)
            {
                PRINTF("Index: %u Dist: %u Edge Target: %u Edge Weight: %u guid: %lu\n", v[j].index, v[j].distance, v[j].edgeList[k].target, v[j].edgeList[k].weight, indexToGuid(v[j].index));
            }
        }
    }
    hiveShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    srand(42);
    graphGuid = hiveMalloc(sizeof(hiveGuid_t)*hiveGetTotalNodes());
    for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
    {
        graphGuid[i] = hiveReserveGuidRoute(HIVE_DB, i % hiveGetTotalNodes());
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        {
            if(hiveIsGuidLocal(graphGuid[i]))
            {
                PRINTF("SIZE ALLOC: %u vert %u edge %u\n", sizeof(vertex) * NUMVERT, sizeof(vertex), sizeof(edge));
                vertex * v = hiveDbCreateWithGuid(graphGuid[i], sizeof(vertex) * NUMVERT);
                for(unsigned int j=0; j<NUMVERT; j++)
                {
                    
                    v[j].index = i*NUMVERT+j;
                    v[j].distance = 0;
                    for(unsigned int k=0; k<NUMEDGE; k++)
                    {
                        v[j].edgeList[k].target = rand() % (NUMVERT*hiveGetTotalNodes());
                        v[j].edgeList[k].weight = (rand() % 25) + 1;
                        
                    }
                }
            }
        }
    }
    
    if(!nodeId && !workerId)
    {
        hiveGuid_t guid = hiveEdtCreate(shutDown, 0, 0, NULL, hiveGetTotalNodes(), NULL);
        for(unsigned int i=0; i<hiveGetTotalNodes(); i++)
        {
            hiveSignalEdt(guid, graphGuid[i], i, DB_MODE_NON_COHERENT_READ);
        }
    }
    
}

int main(int argc, char** argv)
{
    hiveRT(argc, argv);
    return 0;
}