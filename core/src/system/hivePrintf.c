#include "hive.h"
#include "hiveGlobals.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

void PRINTF( const char* format, ... )
{
        va_list arglist;
        printf("[%u] ", hiveGlobalRankId);
        va_start( arglist, format );
        vprintf( format, arglist );
        va_end( arglist );
        fflush(stdout);
}