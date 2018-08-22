#include "arts.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#define NANOSECS 1000000000

u64 artsGetTimeStamp()
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    u64 timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}
