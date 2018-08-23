#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

#define NUMVERT 100
#define NUMEDGE 10

artsGuid_t * graphGuid;

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

artsGuid_t indexToGuid(unsigned int index)
{
    return graphGuid[index/NUMVERT];
}

unsigned int globalIndexToOffset(unsigned int index)
{
    return index % NUMVERT;
}

void visit(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    uint64_t * neighbors = &paramv[1];
    vertex * home = depv[0].ptr;
    unsigned int * distance = &home[index].distance;
    
    for(unsigned int i=1; i<paramc; i++)
    {
        vertex * current = depv[i].ptr;
        unsigned int offset = globalIndexToOffset(paramc[i]);
        unsigned int temp = current[offset].distance + ;
    }
    
    
}

void getDistances(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    vertex * v = depv[0].ptr;
    for(unsigned int i=0; i<NUMVERT; i++)
    {
        uint64_t edges[NUMEDGE+1];
        edges[0] = i;
        for(unsigned int j=0; j<NUMEDGE; j++)
            edges[j+1] = v[i].edgeList[j].target;
        
        artsGuid_t guid = artsEdtCreate(visit, 0, NUMEDGE+1, edges, NUMEDGE+1, NULL);
        
        for(unsigned int j=0; j<NUMEDGE; j++)
            artsSignalEdt(guid, indexToGuid(v[i].edgeList[j].target), j+1, DB_MODE_NON_COHERENT_READ);
        
        artsSignalEdt(guid, indexToGuid(v[i].index), 0, DB_MODE_NON_COHERENT_READ);
    }
    artsShutdown();
}

void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
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
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    srand(42);
    graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        graphGuid[i] = artsReserveGuidRoute(ARTS_DB_READ, i % artsGetTotalNodes());
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            if(artsIsGuidLocal(graphGuid[i]))
            {
                PRINTF("SIZE ALLOC: %u vert %u edge %u\n", sizeof(vertex) * NUMVERT, sizeof(vertex), sizeof(edge));
                vertex * v = artsDbCreateWithGuid(graphGuid[i], sizeof(vertex) * NUMVERT);
                for(unsigned int j=0; j<NUMVERT; j++)
                {
                    
                    v[j].index = i*NUMVERT+j;
                    v[j].distance = 0;
                    for(unsigned int k=0; k<NUMEDGE; k++)
                    {
                        v[j].edgeList[k].target = rand() % (NUMVERT*artsGetTotalNodes());
                        v[j].edgeList[k].weight = (rand() % 25) + 1;
                        
                    }
                }
            }
        }
    }
    
    if(!nodeId && !workerId)
    {
        artsGuid_t guid = artsEdtCreate(shutDown, 0, 0, NULL, artsGetTotalNodes(), NULL);
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            artsSignalEdt(guid, graphGuid[i], i);
        }
    }
    
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}