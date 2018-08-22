#include "arts.h"
#include "artsGlobals.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

void PRINTF( const char* format, ... )
{
        va_list arglist;
        printf("[%u] ", artsGlobalRankId);
        va_start( arglist, format );
        vprintf( format, arglist );
        va_end( arglist );
        fflush(stdout);
}