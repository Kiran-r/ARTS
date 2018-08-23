#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

# if !defined(__APPLE__)
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/prctl.h>

void artsTurnOnCoreDumps()
{
    unsigned int res = prctl(PR_SET_DUMPABLE, 1);

    struct rlimit limit;

    limit.rlim_cur = RLIM_INFINITY ;
    limit.rlim_max = RLIM_INFINITY;
    pid_t pid = getpid();
    if(setrlimit(RLIMIT_CORE, &limit) != 0)
        printf("Failed to force core dumps\n");
}

#else

void artsTurnOnCoreDumps()
{
    printf("Core dumps not supported on OS X.\n");
}

#endif

void artsDebugPrintStack()
{
        void *array[10];
        size_t size = backtrace(array, 10);
        backtrace_symbols_fd(array, size, STDOUT_FILENO);
}

void artsDebugGenerateSegFault()
{
        raise(SIGSEGV);
}
