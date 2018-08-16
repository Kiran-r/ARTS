#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "distributedCsr.h"

graph_csr_distr * createGraphDB(hiveGuid_t guid, u64 index, u64 numPartitions, u64 vertSize, u64 columnSize)
{
    //Allocate space for struct, row vector (vertSize+1), column vector (columnSize)
    u64 space = sizeof(graph_csr_distr) + 
                sizeof(u64) * (numPartitions+1+vertSize+columnSize+1);
    graph_csr_distr * graph = hiveDbCreateWithGuid(guid, space);
    memset(graph, '0', sizeof(space));
    graph->partition = index;
    graph->partSize  = numPartitions;
    graph->vertSize  = vertSize;
    graph->offset    = (u64*)(graph+1);
    graph->row       = graph->offset + numPartitions + 1;
    graph->column    = graph->row + vertSize + 1;
    return graph;
}

void deleteGraph(hiveGuid_t guid)
{
//    hiveDbDestroy(guid);
}

u64 readGraph(char * prefix, hiveGuid_t guid, u64 index, u64 numPartitions, u64 numVert)
{
    u64 vertSize=0 , rowSize=0, columnSize=0;
    char filename[MAXFILENAMESIZE];
    sprintf(filename,"%s.%" PRIu64 "", prefix, index);
    FILE * fp = fopen(filename,"r");
    if(fp)
    {
        if(fscanf(fp, "%" SCNu64 " %" SCNu64 " %" SCNu64 "\n", &vertSize, &rowSize, &columnSize)==3)
        {
            graph_csr_distr * graph = createGraphDB(guid, index, numPartitions, vertSize, columnSize);
            u64 currentRow = 0;
            u64 currentColumn = 0;
            u64 i, j;
            for(i=0; i<vertSize; i++)
            {
                u64 rowCount = 0;
                if(fscanf(fp, "%" SCNu64 "", &rowCount)==1)
                {
                    graph->row[currentRow] = currentColumn;
                    if(rowCount)
                    {
                        for(j=0; j<rowCount; j++)
                        {
                            if(fscanf(fp, "%" SCNu64 "", &graph->column[currentColumn])==1)
                            {
                                currentColumn++;
                            }
                        }
                        currentRow++;
                    }
                }
            }
            graph->row[vertSize] = currentColumn;
            for(i=0, j=0; i<numVert; i+=numVert/numPartitions, j++)
                graph->offset[j]=i;
            graph->offset[numPartitions]=numVert;
        }   
    }
    fclose(fp);
    return vertSize;
}

u64 getStartIndex(graph_csr_distr * graph)
{
    return graph->offset[graph->partition];
}

u64 getEndIndex(graph_csr_distr * graph)
{
    return graph->offset[graph->partition+1];
}

u64 getOffset(graph_csr_distr * graph)
{
    return graph->offset[graph->partition];
}

u64 * getNeighbors(graph_csr_distr * graph, u64 index)
{
    return &graph->column[graph->row[index-getOffset(graph)]];
}

u64 getNumberOfNeighbors(graph_csr_distr * graph, u64 index)
{
    u64 localIndex = index-getOffset(graph);
    return graph->row[localIndex+1] - graph->row[localIndex];
}

u64 getPartition(graph_csr_distr * graph, u64 index)
{
    u64 i;
    for(i=0; i<graph->partSize; i++)
    {
        if(index < graph->offset[i+1])
            return i;
    }
    return INVALID;
}

bool isInCurrentBlock(graph_csr_distr * graph, u64 index)
{
    return (graph->partition==getPartition(graph,index));
}

void printGraph(graph_csr_distr * graph)
{
    u64 start = graph->offset[graph->partition];
    u64 end = graph->offset[graph->partition+1];
    for(u64 i=start; i<end; i++)
    {
        u64 * neighbors = getNeighbors(graph, i);
        u64 numberOfNeighbors = getNumberOfNeighbors(graph, i);
        PRINTF("%" PRIu64 ": %" PRIu64 " {", i, numberOfNeighbors);
        for(u64 j=0; j<numberOfNeighbors; j++)
        {
            printf(" %" PRIu64 "", neighbors[j]);
        }
        printf(" }\n");
    }
}
