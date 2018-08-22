#include <string.h>
#include "arts.h"
#include "artsArrayDb.h"
#include "artsGlobals.h"
#include "artsGuid.h"
#include "artsMalloc.h"
#include "artsDbFunctions.h"
#include "artsEdtFunctions.h"
#include "artsRemoteFunctions.h"
#include "artsDbFunctions.h"
#include "artsTerminationDetection.h"
#include "artsRouteTable.h"
#include "artsOutOfOrder.h"
#include "artsAtomics.h"
#include "artsDebug.h"

artsGuid_t llldddd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("LOOP POLICY!!!\n");
//    artsEdt_t funcPtr = (artsEdt_t)paramv[0];
//    unsigned int stride = paramv[1];
//    unsigned int end = paramv[2];
//    unsigned int start = paramv[3];
//    
//    artsArrayDb_t * array = depv[0].ptr;
//    unsigned int offset = getOffsetFromIndex(array, start);
//    char * raw = depv[0].ptr;
//    
//    for(unsigned int i=start; i<end; i+=stride)
//    {
//        paramv[3] = i;
//        depv[0].ptr = (void*)(&raw[offset]);
//        funcPtr(paramc-3, &paramv[3], 1, depv);
//        offset+=array->elementSize;
//    }
//    depv[0].ptr = (void*)raw;
}