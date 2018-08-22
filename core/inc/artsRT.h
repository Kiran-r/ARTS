#ifndef ARTSRT_H
#define ARTSRT_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsMalloc.h"
#include "artsEdtFunctions.h"
#include "artsDbFunctions.h"
#include "artsGuid.h"
#include "artsUtil.h"
#include "artsTimer.h"
#include "artsArrayDb.h"
#include "artsTerminationDetection.h"
void PRINTF( const char* format, ... );
int artsRT(int argc, char **argv);
#ifdef __cplusplus
}
#endif

#endif /* ARTSRT_H */

