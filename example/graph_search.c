#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include "artsRT.h"
#include "artsGraph.h"
#include "artsTerminationDetection.h"
#include "shadAdapter.h"

unsigned int introStart = 5;

arts_block_dist_t distribution;
csr_graph graph;
char* _file = NULL;
char* _id_file = NULL;
artsGuid_t vertexPropertyMapGuid = NULL_GUID;
artsGuid_t vertexIDMapGuid = NULL_GUID;

u64 startTime;
u64 endTime;

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

artsGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]);

artsGuid_t exitProgram(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
  endTime = artsGetTimeStamp();
  printf("Total execution time: %f s \n", (double)(endTime - startTime)/1000000000.0);
  artsStopIntroShad();
  artsShutdown();
}

artsGuid_t GatherNeighborPropertyVal(u32 paramc, u64 * paramv,
				     u32 depc, artsEdtDep_t depv[]) {
  sourceInfo * srcInfo = depv[depc - 1].ptr;
  vertexProperty * maxWeightedNeighbor = depv[0].ptr;
  for (unsigned int i = 0; i < srcInfo->numNeighbors; i++) {
    vertexProperty * data = depv[i].ptr;
    // TODO: For now, its inefficiently getting both v and id, could have discarded v.
    vertexID * vId = depv[i + srcInfo->numNeighbors].ptr;
    /*For now, just printing in-place*/
//    PRINTF("Seed: %u, Step: %u, Neighbor: %u, neibID: %llu Weight: %f, Visited: %d, Indicator computation: \n", srcInfo->seed, num_steps - srcInfo->step + 1, data->v,vId->id, data->propertyVal, srcInfo->source == data->v ? 1 : 0);
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
    artsGuid_t visitSourceGuid = artsEdtCreate(visitSource, rank, 3, (u64*) & packed_values, 2);
    //        PRINTF("New Edt: %lu Source is located on rank %d Guid: %lu\n", visitSourceGuid, rank, vertexPropertyMapGuid);
    artsSignalEdt(visitSourceGuid, 0, vertexPropertyMapGuid);     
    artsSignalEdt(visitSourceGuid, 1, vertexIDMapGuid);
  }
}

artsGuid_t visitSource(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) { 
//  artsStartIntroShad(introStart);
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
    artsGuid_t dbGuid = artsDbCreate(&ptr, dbSize, ARTS_DB_ONCE_LOCAL);
    sourceInfo * srcInfo = ptr;
    srcInfo->source = source;
    srcInfo->step = nSteps;
    srcInfo->seed = seed;
    srcInfo->numNeighbors = neighbor_cnt;
    // PRINTF("Exploring from Source  %" PRIu64 " steps: %d with neighbors %d\n", source, num_steps + 1 - nSteps, neighbor_cnt);
    // memcpy(&(srcInfo->neighbors), &neighbors, neighbor_cnt * sizeof(vertex));
    /* //... keep filling in */
    artsGuid_t GatherNeighborPropertyValGuid = artsEdtCreate(
						      GatherNeighborPropertyVal,
						      artsGetCurrentNode(), 0,
						      NULL, 2 * neighbor_cnt + 1);
        
    artsSignalEdt(GatherNeighborPropertyValGuid, 2 * neighbor_cnt, dbGuid);
        
    artsArrayDb_t * vertexPropertyMap = depv[0].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex neib = neighbors[i];
      artsGetFromArrayDb(GatherNeighborPropertyValGuid, i, vertexPropertyMap,
			 neib);
    }

    artsArrayDb_t * vertexIDMap = depv[1].ptr;
    for (unsigned int i = 0; i < neighbor_cnt; i++) {
      vertex neib = neighbors[i];
      // PRINTF("Vertex=%llu indexing at %u \n", neib, neighbor_cnt + i);
      artsGetFromArrayDb(GatherNeighborPropertyValGuid, neighbor_cnt + i,
    			 vertexIDMap, neib);
    }
  }
}

artsGuid_t check(u32 paramc, u64 * paramv, u32 depc, artsEdtDep_t depv[]) {
    for (unsigned int i = 0; i < depc; i++) {
        vertexProperty * data = depv[i].ptr;
//        PRINTF("%d %f: %u\n", i, data->v, data->propertyVal);
    }

    artsShutdown();
}

artsGuid_t endVertexIDMapRead(u32 paramc, u64 * paramv,
        u32 depc, artsEdtDep_t depv[]) {
  artsGuid_t exitGuid = artsEdtCreate(exitProgram, 0, 0, NULL, 1);    
  artsInitializeAndStartEpoch(exitGuid, 0);
    
  u64* seeds = (u64*) malloc(sizeof (u64) * num_seeds);

  /*A sanity check that the data is put in properly*/
  /* artsGuid_t edtGuid = artsEdtCreate(check, 0, 0, NULL, distribution.num_vertices); */
  /* for(unsigned int i = 0; i < distribution.num_vertices; i++) */
  /*   artsGetFromArrayDb(edtGuid, i, vertexPropertymap, i); */

  /*Sample seeds*/
  if(fixedSeed > -1)
    {
      seeds[0] = fixedSeed;
    }
  else
    {
      for (int i = 0; i < num_seeds; i++) {
	seeds[i] = rand() % distribution.num_vertices;
//	PRINTF("Seed chosen %d,\n", seeds[i]);
      }
    }
  artsStartIntroShad(introStart);
  startTime = artsGetTimeStamp();
  /*Start walk from each seed in parallel*/
  for (int i = 0; i < num_seeds; i++) {
    vertex source = seeds[i];
    node_t rank = getOwner(source, &distribution);
    // PRINTF("Source is located on rank %d\n", rank);
    /*Spawn an edt at rank that is the owner of current seed vertex*/
    u64 packed_values[3] = {source, num_steps, source};
    artsGuid_t visitSourceGuid = artsEdtCreate(visitSource, rank, 3, (u64*) &packed_values, 2);
    // TODO: why pass vertexpropertguid as an argument?
    artsSignalEdt(visitSourceGuid, 0, vertexPropertyMapGuid);

    artsSignalEdt(visitSourceGuid, 1, vertexIDMapGuid);
  }
}

artsGuid_t endVertexPropertyRead(u32 paramc, u64 * paramv,
        u32 depc, artsEdtDep_t depv[]) {
  
  /*Now read in the vertex ID map*/
  
  //Start an epoch to read in the ID value
  artsGuid_t endVertexIDMapReadEpochGuid
    = artsEdtCreate(endVertexIDMapRead, 0, 0, NULL, 2);
  
  // TODO: Is the following line necessary ?
  //Signal the ID map guid
  artsSignalEdt(endVertexIDMapReadEpochGuid, 1, vertexIDMapGuid);

  //Start the epoch
  artsInitializeAndStartEpoch(endVertexIDMapReadEpochGuid, 0);

  // Allocate vertex ID map and populate it from node 0
  artsArrayDb_t * vertexIDMap = artsNewArrayDbWithGuid(vertexIDMapGuid,
						       sizeof (vertexID), 
						       distribution.num_vertices);

  //Read in property file
  PRINTF("[INFO] Reading in and constructing the vertex id map ...\n");
  FILE *file = fopen(_id_file, "r");
  PRINTF("File to be opened %s\n", _id_file);
  if (file == NULL) {
    PRINTF("[ERROR] File containing vertex ids  can't be open -- %s", _file);
    artsShutdown();
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
	// PRINTF("Vertex=%llu ", vertex);
	++i;
      } else if (i == 1) { // id
	id = atoll(token);
	// PRINTF("id=%llu\n", id);
	i = 0;
      }
      token = strtok(NULL, " ");
    }
    vertexID vIDInfo = {.v = vertex, .id = id};

    artsPutInArrayDb(&vIDInfo, NULL_GUID, 0, vertexIDMap, index);
    index++;
  }
  fclose(file);
}

void initPerNode(unsigned int nodeId, int argc, char** argv) {

  //This is the dbGuid we will need to aquire to do gets and puts to the score property arrayDb
  vertexPropertyMapGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
  vertexIDMapGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);

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
    artsGuid_t endVertexPropertyReadEpochGuid
      = artsEdtCreate(endVertexPropertyRead, 0, 0, NULL, 2);
        
    //Signal the property map guid
    artsSignalEdt(endVertexPropertyReadEpochGuid, 1, vertexPropertyMapGuid);

    //Start the epoch
    artsInitializeAndStartEpoch(endVertexPropertyReadEpochGuid, 0);

    // Allocate vertex property map and populate it from node 0
    artsArrayDb_t * vertexPropertyMap = artsNewArrayDbWithGuid(
					       vertexPropertyMapGuid,
					       sizeof (vertexProperty), 
					       distribution.num_vertices);

    //Read in property file
    PRINTF("[INFO] Reading in and constructing the vertex property map ...\n");
    FILE *file = fopen(_file, "r");
    PRINTF("File to be opened %s\n", _file);
    if (file == NULL) {
      PRINTF("[ERROR] File containing property value can't be open -- %s", _file);
      artsShutdown();
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

      artsPutInArrayDb(&vPropVal, NULL_GUID, 0, vertexPropertyMap, index);
      index++;
    }
    fclose(file);
  }
}

int main(int argc, char** argv) {
    artsRT(argc, argv);
    return 0;
}

