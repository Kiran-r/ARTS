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
char* _file = NULL;
char* _id_file = NULL;
hiveGuid_t vertexPropertyMapGuid = NULL_GUID;
hiveGuid_t vertexIDMapGuid = NULL_GUID;

/*Default values as in python code*/
int num_seeds = 25;
int num_steps = 1500;

int fixedSeed = -1;

typedef struct {
    vertex v;
    double propertyVal;
} vertexProperty;

typedef struct {
    vertex v;
    vertex id;
} vertexID;

typedef struct {
    vertex source;
    unsigned int step;
    unsigned int numNeighbors;
    vertex seed;
    // vertex neighbors[];
} sourceInfo;

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]);

hiveGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    hiveShutdown();
}

hiveGuid_t GatherNeighborPropertyVal(u32 paramc, u64 * paramv,
				     u32 depc, hiveEdtDep_t depv[]) {
  sourceInfo * srcInfo = depv[depc - 1].ptr;
  vertexProperty * maxWeightedNeighbor = depv[0].ptr;
  for (unsigned int i = 0; i < srcInfo->numNeighbors; i++) {
    vertexProperty * data = depv[i].ptr;
    // TODO: For now, its inefficiently getting both v and id, could have discarded v.
    vertexID * vId = depv[i + srcInfo->numNeighbors].ptr;
    /*For now, just printing in-place*/
    PRINTF("Seed: %u, Step: %u, Neighbor: %u, neibID: %llu Weight: %f, Visited: %d, Indicator computation: \n", srcInfo->seed, num_steps - srcInfo->step + 1, data->v,vId->id, data->propertyVal, srcInfo->source == data->v ? 1 : 0);
    /*For now we are doing in-place max-weighted sampling for next source*/
    if (data->propertyVal > maxWeightedNeighbor->propertyVal) {
      maxWeightedNeighbor->v = data->v;
      maxWeightedNeighbor->propertyVal = data->propertyVal;
    }
  }

  /*spawn next step*/
  if (srcInfo->step > 0) {
    vertex source = maxWeightedNeighbor->v;
    node_t rank = getOwner(source, &distribution);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    u64 packed_values[3] = {source, srcInfo->step - 1, srcInfo->seed};
    hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 3, (u64*) & packed_values, 1);
    //        PRINTF("New Edt: %lu Source is located on rank %d Guid: %lu\n", visitSourceGuid, rank, vertexPropertyMapGuid);
    // hiveGuid_t vertexPropertyMapGuid= depv[2 *  srcInfo->numNeighbors + 1].guid;
    // hiveGuid_t vertexIDMapGuid= depv[2 *  srcInfo->numNeighbors + 2].guid;
    // hiveArrayDb_t * vertexIDMap = depv[2 *  srcInfo->numNeighbors + 2].ptr;
    hiveSignalEdt(visitSourceGuid, vertexPropertyMapGuid, 0, DB_MODE_PIN);     
    hiveSignalEdt(visitSourceGuid, vertexIDMapGuid, 1, DB_MODE_PIN);
  }
}

hiveGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) { 
  vertex* neighbors = NULL;
  u64 neighbor_cnt = 0;    
  vertex source = (vertex) paramv[0];
  int nSteps = (int) paramv[1];
  vertex seed = (vertex) paramv[2];
  //    PRINTF("Current Source  %" PRIu64 "\n", source);

  getNeighbors(&graph, source, &neighbors, &neighbor_cnt);
  if (neighbor_cnt) {
    unsigned int dbSize = sizeof (sourceInfo); // + neighbor_cnt * sizeof(vertex);
    void * ptr = NULL;
    hiveGuid_t dbGuid = hiveDbCreate(&ptr, dbSize, false);
    sourceInfo * srcInfo = ptr;
    srcInfo->source = source;
    srcInfo->step = nSteps;
    srcInfo->seed = seed;
    srcInfo->numNeighbors = neighbor_cnt;
    PRINTF("Exploring from Source  %" PRIu64 " steps: %d with neighbors %d\n", source, num_steps + 1 - nSteps, neighbor_cnt);
    // memcpy(&(srcInfo->neighbors), &neighbors, neighbor_cnt * sizeof(vertex));
    /* //... keep filling in */
    // TODO: Do the slots have to have same datasize?
    hiveGuid_t GatherNeighborPropertyValGuid = hiveEdtCreate(
						      GatherNeighborPropertyVal,
						      hiveGetCurrentNode(), 0,
						      NULL, 2 * neighbor_cnt + 1);
        
    hiveSignalEdt(GatherNeighborPropertyValGuid, dbGuid, 2 * neighbor_cnt, DB_MODE_ONCE_LOCAL);
    // hiveSignalEdt(GatherNeighborPropertyValGuid, depv[0].guid, 2 * neighbor_cnt + 1, DB_MODE_ONCE_LOCAL);
    // hiveSignalEdt(GatherNeighborPropertyValGuid, depv[1].guid, 2 * neighbor_cnt + 2, DB_MODE_ONCE_LOCAL);
        
    hiveArrayDb_t * vertexPropertyMap = depv[0].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex neib = neighbors[i];
      hiveGetFromArrayDb(GatherNeighborPropertyValGuid, i, vertexPropertyMap,
			 neib);
    }

    hiveArrayDb_t * vertexIDMap = depv[1].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex neib = neighbors[i];
      PRINTF("Vertex=%llu indexing at %u \n", neib, neighbor_cnt + i);
      hiveGetFromArrayDb(GatherNeighborPropertyValGuid, neighbor_cnt + i,
    			 vertexIDMap, neib);
    }
  }
}

hiveGuid_t check(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[]) {
    for (unsigned int i = 0; i < depc; i++) {
        vertexProperty * data = depv[i].ptr;
        PRINTF("%d %f: %u\n", i, data->v, data->propertyVal);
    }

    hiveShutdown();
}

hiveGuid_t endVertexIDMapRead(u32 paramc, u64 * paramv,
        u32 depc, hiveEdtDep_t depv[]) {
  hiveGuid_t exitGuid = hiveEdtCreate(exitProgram, 0, 0, NULL, 1);    
  hiveInitializeAndStartEpoch(exitGuid, 0);
    
  u64* seeds = (u64*) malloc(sizeof (u64) * num_seeds);

  /*A sanity check that the data is put in properly*/
  /* hiveGuid_t edtGuid = hiveEdtCreate(check, 0, 0, NULL, distribution.num_vertices); */
  /* for(unsigned int i = 0; i < distribution.num_vertices; i++) */
  /*   hiveGetFromArrayDb(edtGuid, i, vertexPropertymap, i); */

  /*Sample seeds*/
  if(fixedSeed > -1)
    {
      seeds[0] = fixedSeed;
    }
  else
    {
      for (int i = 0; i < num_seeds; i++) {
	seeds[i] = rand() % 100;//distribution.num_vertices;
	PRINTF("Seed chosen %d,\n", seeds[i]);
      }
    }
    
  /*Start walk from each seed in parallel*/
  for (int i = 0; i < num_seeds; i++) {
    vertex source = seeds[i];
    node_t rank = getOwner(source, &distribution);
    PRINTF("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    u64 packed_values[3] = {source, num_steps, source};
    hiveGuid_t visitSourceGuid = hiveEdtCreate(visitSource, rank, 3, (u64*) &packed_values, 2);
    // TODO: why pass vertexpropertguid as an argument?
    hiveSignalEdt(visitSourceGuid, vertexPropertyMapGuid, 0, DB_MODE_PIN);

    hiveSignalEdt(visitSourceGuid, vertexIDMapGuid, 1, DB_MODE_PIN);
  }
}

hiveGuid_t endVertexPropertyRead(u32 paramc, u64 * paramv,
        u32 depc, hiveEdtDep_t depv[]) {
  
  /*Now read in the vertex ID map*/
  
  //Start an epoch to read in the ID value
  hiveGuid_t endVertexIDMapReadEpochGuid
    = hiveEdtCreate(endVertexIDMapRead, 0, 0, NULL, 2);
  
  // TODO: Is the following line necessary ?
  //Signal the ID map guid
  hiveSignalEdt(endVertexIDMapReadEpochGuid, vertexIDMapGuid, 1, DB_MODE_PIN);

  //Start the epoch
  hiveInitializeAndStartEpoch(endVertexIDMapReadEpochGuid, 0);

  // Allocate vertex ID map and populate it from node 0
  hiveArrayDb_t * vertexIDMap = hiveNewArrayDbWithGuid(vertexIDMapGuid,
						       sizeof (vertexID), 
						       distribution.num_vertices);

  //Read in property file
  PRINTF("[INFO] Reading in and constructing the vertex id map ...\n");
  FILE *file = fopen(_id_file, "r");
  PRINTF("File to be opened %s\n", _id_file);
  if (file == NULL) {
    PRINTF("[ERROR] File containing vertex ids  can't be open -- %s", _file);
    hiveShutdown();
  }

  PRINTF("Started reading the vertex ids file..\n");

  char str[MAXCHAR];
  u64 index = 0;
  while (fgets(str, MAXCHAR, file) != NULL) {
    graph_sz_t vertex;
    graph_sz_t id;
    char* token = strtok(str, "\t");
    int i = 0;
    while (token != NULL) {
      if (i == 0) { // vertex
	vertex = atoll(token);
	PRINTF("Vertex=%llu ", vertex);
	++i;
      } else if (i == 1) { // id
	id = atoll(token);
	PRINTF("id=%llu\n", id);
	i = 0;
      }
      token = strtok(NULL, " ");
    }
    vertexID vIDInfo = {.v = vertex, .id = id};

    hivePutInArrayDb(&vIDInfo, NULL_GUID, 0, vertexIDMap, index);
    index++;
  }
  fclose(file);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {

  //This is the dbGuid we will need to aquire to do gets and puts to the score property arrayDb
  vertexPropertyMapGuid = hiveReserveGuidRoute(HIVE_DB, 0);
  vertexIDMapGuid = hiveReserveGuidRoute(HIVE_DB, 0);

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
  if (!nodeId && !workerId) {
    for (int i = 0; i < argc; ++i) {
      if (strcmp("--propertyfile", argv[i]) == 0) {
	_file = argv[i + 1];
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

    for (int i = 0; i < argc; ++i) {
      if (strcmp("--idfile", argv[i]) == 0) {
	_id_file = argv[i + 1];  
      }
    }

    /* // How many seeds   */
    /* for (int i = 0; i < argc; ++i) { */
    /*     if (strcmp("--fixed", argv[i]) == 0) { */
    /*         sscanf(argv[i + 1], "%d", &fixedSeed); */
    /*         num_seeds = 1; */
    /*     } */
    /* } */
        
    //Start an epoch to read in the property value
    hiveGuid_t endVertexPropertyReadEpochGuid
      = hiveEdtCreate(endVertexPropertyRead, 0, 0, NULL, 2);
        
    //Signal the property map guid
    hiveSignalEdt(endVertexPropertyReadEpochGuid, vertexPropertyMapGuid, 1, DB_MODE_PIN);

    //Start the epoch
    hiveInitializeAndStartEpoch(endVertexPropertyReadEpochGuid, 0);

    // Allocate vertex property map and populate it from node 0
    hiveArrayDb_t * vertexPropertyMap = hiveNewArrayDbWithGuid(
					       vertexPropertyMapGuid,
					       sizeof (vertexProperty), 
					       distribution.num_vertices);

    //Read in property file
    PRINTF("[INFO] Reading in and constructing the vertex property map ...\n");
    FILE *file = fopen(_file, "r");
    PRINTF("File to be opened %s\n", _file);
    if (file == NULL) {
      PRINTF("[ERROR] File containing property value can't be open -- %s", _file);
      hiveShutdown();
    }

    PRINTF("Started reading the vertex property file..\n");
    char str[MAXCHAR];
    u64 index = 0;
    while (fgets(str, MAXCHAR, file) != NULL) {
      graph_sz_t vertex;
      double vPropertyVal;
      char* token = strtok(str, "\t");
      int i = 0;
      while (token != NULL) {
	if (i == 0) { // vertex
	  vertex = atoll(token);
	  // PRINTF("Vertex=%llu ", vertex);
	  ++i;
	} else if (i == 1) { // property
	  vPropertyVal = atof(token);
	  // PRINTF("propval=%f\n", vPropertyVal);
	  i = 0;
	}
	token = strtok(NULL, " ");
      }
      vertexProperty vPropVal = {.v = vertex, .propertyVal = vPropertyVal};

      hivePutInArrayDb(&vPropVal, NULL_GUID, 0, vertexPropertyMap, index);
      index++;
    }
    fclose(file);
  }
}

int main(int argc, char** argv) {
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
