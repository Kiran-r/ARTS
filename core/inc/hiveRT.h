#ifndef HIVERT_H
#define HIVERT_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hive.h"
#include "hiveMalloc.h"
#include "hiveEdtFunctions.h"
#include "hiveDbFunctions.h"
#include "hiveGuid.h"
#include "hiveUtil.h"
#include "hiveTimer.h"
#include "hiveArrayDb.h"
#include "hiveTerminationDetection.h"
void PRINTF( const char* format, ... );
int hiveRT(int argc, char **argv);
#ifdef __cplusplus
}
#endif

#endif /* HIVERT_H */

