#ifndef DISTRIBUTEDCSR_H
#define	DISTRIBUTEDCSR_H

#define INVALID (u64)-1
#define MAXFILENAMESIZE 128    
    
#include "hiveRT.h"
    
    typedef struct 
    {
        u64 partition;
        u64 partSize;
        u64 vertSize;
        u64 * offset;
        u64 * row;
        u64 * column;
    } graph_csr_distr;

void createGraphDB(hiveGuid_t * guid, graph_csr_distr ** ptr, u64 index, u64 numPartitions, u64 vertSize, u64 columnSize);
void deleteGraph(hiveGuid_t guid);
u64 getStartIndex(graph_csr_distr * graph);
u64 getEndIndex(graph_csr_distr * graph);
u64 getOffset(graph_csr_distr * graph);
u64 * getNeighbors(graph_csr_distr * graph, u64 index);
u64 getNumberOfNeighbors(graph_csr_distr * graph, u64 index);
u64 getPartition(graph_csr_distr * graph, u64 index);
bool isInCurrentBlock(graph_csr_distr * graph, u64 index);
void readGraph(char * prefix, hiveGuid_t * guid, graph_csr_distr ** ptr, u64 index, u64 numPartitions, u64 numVert);
void printGraph(graph_csr_distr * graph);

#endif	/* DISTRIBUTEDCSR_H */

