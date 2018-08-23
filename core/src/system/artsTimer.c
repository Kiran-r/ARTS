#include <time.h>
#include "arts.h"

#define NANOSECS 1000000000

uint64_t artsGetTimeStamp()
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    uint64_t timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}
