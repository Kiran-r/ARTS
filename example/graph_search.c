#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "hiveRT.h"
#include "hiveGraph.h"
#include "hiveTerminationDetection.h"

hive_block_dist_t distribution;
csr_graph graph;
hiveArrayDb_t * vertexPropertymap = NULL;
char* _file = NULL;
hiveGuid_t printRandomWalkInfoGuid = NULL_GUID;

/*Default values as in python code*/
int num_seeds = 25;
int num_steps = 1500;

typedef struct {
  vertex v;
  double propertyVal;
} vertexProperty;

typedef struct 
{
  vertex source;
  unsigned int step;
  unsigned int numNeighbors;
  vertex seed;
  // vertex neighbors[];
} sourceInfo;

typedef struct 
{
  vertex source;
  unsigned int step;
  unsigned int numNeighbors;
  vertex seed;
  vertexProperty neighborProperty[];
} randomWalkInfo;

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

/* hiveGuid_t printRandomWalkInfo(u32 paramc, u64 * paramv,  */
/* 				     u32 depc, hiveEdtDep_t depv[]) { */
/*   PRINTF("Printing Random Walk info\n"); */
/*   for(unsigned int i = 0; i < depc; i++) { */
/*     randomWalkInfo * data = depv[i].ptr; */
/*     for(unsigned int j = 0; j < data->numNeighbors; j++) { */
/*       PRINTF("Seed: %u, Step: %u, Neighbor: %u, Weight: %u, Visited: %d, Indicator computation: \n", data->seed, data->step, data->neighborProperty[j].v, data->neighborProperty[j].propertyVal,  (data->source == data->neighborProperty[j].v) ); */
/*     } */
/*   } */

/*   hiveShutdown(); */

/* } */

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    hiveShutdown();
}

hiveGuid_t printRandomWalkInfo(u32 paramc, u64 * paramv, 
				     u32 depc, hiveEdtDep_t depv[]) {
  PRINTF("Printing Random Walk info\n");
  for(unsigned int i = 0; i < depc; i++) {
    randomWalkInfo * data = depv[i].ptr;
    for(unsigned int j = 0; j < data->numNeighbors; j++) {
      PRINTF("Seed: %u, Step: %u, Neighbor: %u, Weight: %u, Visited: %d, Indicator computation: \n", data->seed, data->step, data->neighborProperty[j].v, data->neighborProperty[j].propertyVal,  (data->source == data->neighborProperty[j].v) );
    }
  }
  // hiveShutdown();
}

hiveGuid_t GatherNeighborPropertyVal(u32 paramc, u64 * paramv, 
				     u32 depc, hiveEdtDep_t depv[]) {
  sourceInfo* srcInfo = depv[depc-1].ptr; 
  vertexProperty * maxWeightedNeighbor = depv[0].ptr;
  for(unsigned int i = 0; i < depc - 1; i++) {
    vertexProperty * data = depv[i].ptr;
    /*For now, just printing in-place*/
    PRINTF("Seed: %u, Step: %u, Neighbor: %u, Weight: %f, Visited: %d, Indicator computation: \n", srcInfo->seed, srcInfo->step,  data->v, data->propertyVal,  srcInfo->source == data->v ? 1: 0) ;
    /*For now we are doing in-place max-weighted sampling for next source*/
    if (data->propertyVal > maxWeightedNeighbor->propertyVal) {
      maxWeightedNeighbor->v = data->v;
      maxWeightedNeighbor->propertyVal = data->propertyVal;
    }
  }

  /* unsigned int dbSize = sizeof(randomWalkInfo) +  (depc - 1) * sizeof(vertexProperty); */
  /* void * ptr = NULL; */
  /* hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false); */
  
  /* randomWalkInfo * randWalkInfo = ptr; */
  
  /* randWalkInfo->source = srcInfo->source; */
  /* randWalkInfo->step = srcInfo->step; */
  /* randWalkInfo->numNeighbors = depc - 2; */
  /* randWalkInfo->seed = srcInfo->seed; */
  /* memcpy(&(randWalkInfo->neighborProperty), &depv, (depc - 1) * sizeof(vertexProperty));   */
    
  // u32 slotNo = srcInfo->seed * num_steps - srcInfo->step;  
  // TODO: Print info , sample new source, and start next step
  // hiveSignalEdt(printRandomWalkInfoGuid, dbGuid, slotNo, DB_MODE_NON_COHERENT_READ);
  /*spawn next step*/
  if (srcInfo->step > 0) {
    vertex source = maxWeightedNeighbor->v;
    node_t rank = getOwner(source, &distribution);
    printf("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    u64 packed_values[3] = {source, srcInfo->step - 1,  srcInfo->seed};
    hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 3, 
					       (u64*) &packed_values, 0);
  }
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
  // hiveShutdown();
  vertex* neighbors = NULL; 
  u64 neighbor_cnt = 0; 
  vertex source = (vertex) paramv[0];
  int nSteps = (int) paramv[1];
  vertex seed = (vertex) paramv[2];
  printf("Current Source  %" PRIu64 "\n", source); 
  // hiveShutdown();
  
  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);

  // TODO: what if no neighbors? if (neighbor_cnt) 
  
  if (neighbor_cnt) {
    unsigned int dbSize = sizeof(sourceInfo); // + neighbor_cnt * sizeof(vertex);
    void * ptr = NULL;
    hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
    sourceInfo * srcInfo = ptr;
    srcInfo->source = source;
    srcInfo->step = nSteps;
    srcInfo->seed = seed;
    printf("Exploring from Source  %" PRIu64 " steps: %d \n", source, num_steps + 1 - nSteps); 
    // memcpy(&(srcInfo->neighbors), &neighbors, neighbor_cnt * sizeof(vertex));
    /* //... keep filling in */
  
    hiveGuid_t GatherNeighborPropertyValGuid = hiveEdtCreate(
					         GatherNeighborPropertyVal,
						 hiveGetCurrentNode(), 0,
						 NULL, neighbor_cnt + 1);
  
    hiveSignalEdt(GatherNeighborPropertyValGuid, dbGuid, neighbor_cnt, DB_MODE_NON_COHERENT_READ);
  
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex neib = neighbors[i];
      hiveGetFromArrayDb(GatherNeighborPropertyValGuid, i, vertexPropertymap,
			 neib);
    }
  }
}

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{    
    for(unsigned int i=0; i<depc; i++)
    {
        vertexProperty * data = depv[i].ptr;
        PRINTF("%d %f: %u\n", i, data->v, data->propertyVal);
    }
    
    hiveShutdown();
}

hiveGuid_t startRandomWalk(u32 paramc, u64 * paramv, 
				 u32 depc, hiveEdtDep_t depv[]) {
  u64* seeds = (u64*) malloc(sizeof(u64) * num_seeds);

  /*A sanity check that the data is put in properly*/
  /* hiveGuid_t edtGuid = hiveEdtCreate(check, 0, 0, NULL, distribution.num_vertices); */
  /* for(unsigned int i = 0; i < distribution.num_vertices; i++) */
  /*   hiveGetFromArrayDb(edtGuid, i, vertexPropertymap, i); */
  
  /*Sample seeds*/
  printf("Num seeds: %d \n", num_seeds);
  for (int i = 0; i < num_seeds; i++) {
    seeds[i] = rand() % distribution.num_vertices;
    printf("seed chosen %d,\n", seeds[i]);
  }
  // hiveShutdown();
  /* Create an EDT for printing all info at the end. */
  // printRandomWalkInfoGuid = hiveEdtCreate(printRandomWalkInfo, 0, 0, NULL, num_seeds * num_steps); 

  /*Start walk from each seed in parallel*/
  for (int i = 0; i < num_seeds; i++) {
    vertex source = seeds[i];
    node_t rank = getOwner(source, &distribution);
    printf("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    u64 packed_values[3] = {source, num_steps, source};
    hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 3, 
					       (u64*) &packed_values, 0);
    /*TODO: check: Since we are passing along stepcount as an argument, 
      we pass the source as the depv*/
    //  hiveSignalEdt(visitSourceGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
  }

}


hiveGuid_t endVertexPropertyRead(u32 paramc, u64 * paramv, 
				 u32 depc, hiveEdtDep_t depv[]) {    
  // TODO: check how many times to signal the exit EDT
  hiveGuid_t exitGuid = hiveEdtCreate(exitProgram, 0, 0, NULL, num_seeds);
  hiveGuid_t startGuid = hiveEdtCreate(startRandomWalk, 0, 0, NULL, 1);
  hiveInitializeEpoch(startGuid, exitGuid, 0);
  
  //Start
  hiveSignalEdt(startGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
}

hiveGuid_t startVertexPropertyRead(u32 paramc, u64 * paramv, 
				   u32 depc, hiveEdtDep_t depv[]) {
  // Allocate vertex property map and populate it from node 0
  hiveNewArrayDb(&vertexPropertymap, sizeof(vertexProperty), 
		 getBlockSize(&distribution), hiveGetTotalNodes());

  PRINTF("[INFO] Reading in and constructing the vertex property map ...\n");
  FILE *file = fopen(_file, "r");
  PRINTF("File to be opened %s\n", _file);
  if (file == NULL) {
    PRINTF("[ERROR] File containing property value can't be open -- %s", _file);
    // hiveShutdown();
  }

  PRINTF("Started reading the vertex property file..\n");
  char str[MAXCHAR];
  u64 index = 0;
  while (fgets(str, MAXCHAR, file) != NULL) {
    //      PRINTF("here \n");
    graph_sz_t vertex;
    double vPropertyVal;
    char* token = strtok(str, "\t");
    int i = 0;
    while(token != NULL) {
      // printf("Iteration %d", it);
      if (i == 0) { // vertex
	vertex = atoll(token);
	printf("Vertex=%llu ", vertex);
	++i;
      } else if (i == 1) { // property
	vPropertyVal = atof(token);
	printf("propval=%f\n", vPropertyVal);
	i = 0;
      }

      token = strtok(NULL, " ");
    }
    vertexProperty vPropVal = { .v = vertex, .propertyVal = vPropertyVal}; 
         
    hivePutInArrayDb(&vPropVal, NULL_GUID, 0, vertexPropertymap, index);      
    index++;
  }
  fclose(file);
}


void initPerNode(unsigned int nodeId, int argc, char** argv) {
     
  // distribution must be initialized in initPerNode
  initBlockDistributionWithCmdLineArgs(&distribution, 
                                       argc, argv);
  // read the edgelist and construct the graph
  loadGraphUsingCmdLineArgs(&graph,
			    &distribution,
			    argc,
			    argv);

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, 
		   int argc, char** argv) {
  if (!nodeId  && !workerId) {
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--propertyfile", argv[i]) == 0) {
  	_file = argv[i+1];
      }
    }

    // How many seeds  
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--num-seeds", argv[i]) == 0) {
	sscanf(argv[i + 1], "%d", &num_seeds);
      }
    }

    // How many steps  
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--num-steps", argv[i]) == 0) {
	sscanf(argv[i + 1], "%d", &num_steps);
      }
    }
 
    /*Start an epoch to read in the property value*/
    hiveGuid_t endVertexPropertyReadEpochGuid 
      = hiveEdtCreate(endVertexPropertyRead, 0, 0, NULL, 1);
    hiveGuid_t startVertexPropertyReadEpochGuid 
      = hiveEdtCreate(startVertexPropertyRead, 0, 0, NULL, 1);
    hiveInitializeEpoch(startVertexPropertyReadEpochGuid, 
			endVertexPropertyReadEpochGuid, 0);
    hiveSignalEdt(startVertexPropertyReadEpochGuid, 
		  NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
  }
}


int main(int argc, char** argv)
{
  hiveRT(argc, argv);
  return 0;
}


 /*Sample seeds: data structures?*/
  /*Sampling and strting the random walk should only happen from node 0*/
  /*TODO: Accumulate in hiveEdgeVector?*/
  /*or hiveDb? is hiveDb more heavy-weight?*/

/*   vertex seed = rand() % distribution.num_vertices; */
/*   u64 indicator_computations = 0; */
  
/*   /\*for each seed*\/ */
/*   for (int seed_no = 0; seed_no < num_seeds; seed_no++) { */
/*     vertex source = seed; */
/*     /\*TODO: For each step*\/ */
/*     for (int step = 0; step < num_steps; step++) { */
/*       /\*Get the neighbors*\/ */
/*       vertex* neighbors = NULL; */
/*       u64 neighbor_cnt = 0; */
/*       double* weights = NULL; */
/*       /\*Check whether the seed vertex belongs to the current node*\/ */
/*       if (getOwner(seed, &distribution) ==  hiveGetCurrentNode()) { */
/* 	/\*If the current locality is the owner of the seed, get the 	  neighbors' scores*\/ */
/* 	getNeighbors(&graph, seed, &neighbors, &neighbor_cnt); */
/* 	weights = (double*) hiveMalloc(sizeof(double)); */
/* 	for(u64 i=0; i < neighbor_cnt; ++i) { */
/* 	  /\*Get the scores of the neighbors*\/ */
/* 	  //TODO: copy locally from global array */
/* 	  vertex u = neighbors[i]; */
/* 	  hiveGetFromArrayDb(0, 0, array, u); */
/* 	} */
/* 	indicator_computations += neighbor_cnt; */
/* 	/\*Sample a neighbour based on one of the strategies: */
/* 	  random, argmax, argmin, weighted_random*\/ */
/* 	/\*TODO: Implement sampling strategies*\/ */
/* 	vertex source = sample(); */
/* 	/\*for each neighbor, print the following information:*\/ */
/* 	for(u64 i=0; i < neighbor_cnt; ++i) { */
/* 	  /\*TODO: *\/ */
/* 	  /\*  for neib_id, (neib, weight) in enumerate(zip(neibs, weights)): */
/*                 yield { */
/*                     "seed"    : int(seed), */
/*                     "step"    : int(step), */
/*                     "neib_id" : int(neib_id), */
/*                     "neib"    : int(neib), */
/*                     "weight"  : float(weight), */
/*                     "visited" : int(neib == source), */
/*                     "indicator_computations" : int(indicator_computations), */
/*                 } */

/* *\/ */
/* 	} */
 
/*       } */
/*     } */
/*   } */
