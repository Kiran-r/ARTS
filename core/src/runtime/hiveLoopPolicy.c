#include <string.h>
#include "hive.h"
#include "hiveArrayDb.h"
#include "hiveGlobals.h"
#include "hiveGuid.h"
#include "hiveMalloc.h"
#include "hiveDbFunctions.h"
#include "hiveEdtFunctions.h"
#include "hiveRemoteFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveTerminationDetection.h"
#include "hiveRouteTable.h"
#include "hiveOutOfOrder.h"
#include "hiveAtomics.h"
#include "hiveDebug.h"

hiveGuid_t llldddd(u32 paramc, u64 * paramv, u32 depc, hiveEdtDep_t depv[])
{
    PRINTF("LOOP POLICY!!!\n");
//    hiveEdt_t funcPtr = (hiveEdt_t)paramv[0];
//    unsigned int stride = paramv[1];
//    unsigned int end = paramv[2];
//    unsigned int start = paramv[3];
//    
//    hiveArrayDb_t * array = depv[0].ptr;
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