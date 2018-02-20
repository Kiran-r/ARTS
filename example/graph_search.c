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

/*Default values as in python code*/
int num_seeds = 25;
int num_steps = 1500;

typedef struct {
  vertex v;
  double propertyVal;
} vertexProperty;

hiveGuid_t GatherNeighborPropertyVal(u32 paramc, u64 * paramv, 
				     u32 depc, hiveEdtDep_t depv[]) {
  for(unsigned int i = 0; i < depc; i++) {
    unsigned int * data = depv[i].ptr;
    PRINTF("%u: %u\n", i, *data);
  }
  // TODO: Print info , sample new source, and start next step
  hiveShutdown();
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
  // hiveShutdown();
  vertex* neighbors = NULL; 
  u64 neighbor_cnt = 0; 
  unsigned int * source = depv[0].ptr; // TODO: assuming vertex type is 
  // of type unsigned int
  getNeighbors(&graph, *source, &neighbors, &neighbor_cnt);
  /* TODO: I need to pass in several things to the continuation 
     for printing: source, # steps, actual neighbor ids, 
     indicator_computations etc. What to do about them? */
  hiveGuid_t GatherNeighborPropertyValGuid = hiveEdtCreate(
                                             GatherNeighborPropertyVal, 
                                             hiveGetCurrentNode(), 0, 
					     NULL, neighbor_cnt);
  for (unsigned int i = 0; i < neighbor_cnt; i++) {
    vertex neib = neighbors[i];
    hiveGetFromArrayDb(GatherNeighborPropertyValGuid, i, vertexPropertymap,
		       neib);
  }
}

hiveGuid_t endVertexPropertyRead(u32 paramc, u64 * paramv, 
				 u32 depc, hiveEdtDep_t depv[]) {    
 
  int* seeds = (int*) malloc(sizeof(int) * num_seeds);
  
  /*Sample seeds*/
  for (int i = 0; i < num_seeds; i++) {
    seeds[i] = rand() % distribution.num_vertices;
    printf("seed chosen %d,\n", seeds[i]);
  }

  /*Start walk from each seed in parallel*/
  for (int i = 0; i < num_seeds; i++) {
    vertex source = seeds[i];
    node_t rank = getOwner(source, &distribution);
    printf("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 1, 
					       (u64*) &num_steps, 1);
    /*TODO: check: Since we are passing along stepcount as an argument, 
      we pass the source as the depv*/
    hiveSignalEdt(visitSourceGuid, NULL_GUID, 0, DB_MODE_SINGLE_VALUE);
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
  // for(unsigned int i = 0; i < index; i++)
    // TODO: hiveGetFromArrayDb interface why EDT?
    //hiveGetFromArrayDb(0, 0, array, i);

  // if (!nodeId) 
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
	sscanf(argv[i + 1], "%" SCNu64, &num_seeds);
      }
    }

    // How many steps  
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--num-stepss", argv[i]) == 0) {
	sscanf(argv[i + 1], "%" SCNu64, &num_steps);
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
