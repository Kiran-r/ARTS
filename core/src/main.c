#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include "hive.h"
#include "hiveThreads.h"
#include "hiveConfig.h"
#include "hiveGlobals.h"
#include "hiveRemote.h"
#include "hiveGuid.h"
#include "hiveRuntime.h"
#include "hiveMalloc.h"
#include "hiveRemoteLauncher.h"
#include "hiveIntrospection.h"
#include <sys/resource.h>
//#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>

extern struct hiveConfig * config;

int mainArgc = 0;
char ** mainArgv = NULL;

void turnOnCoreDumps()
{
#if !defined(__APPLE__)
    unsigned int res = prctl(PR_SET_DUMPABLE, 1);

    struct rlimit limit;

    limit.rlim_cur = RLIM_INFINITY ;
    limit.rlim_max = RLIM_INFINITY;
    pid_t pid = getpid();
    if (setrlimit(RLIMIT_CORE, &limit) != 0)
        ONCE_PRINTF("Failed to force core dumps\n");
    else
        ONCE_PRINTF("Core dumps forced on\n");
#else
    ONCE_PRINTF("Core dumps not supported on OS X.\n");
#endif
}

int hiveRT(int argc, char **argv)
{
    mainArgc = argc;
    mainArgv = argv;
    hiveRemoteTryToBecomePrinter();
    config = hiveConfigLoad(0, NULL, NULL);

    if(config->coreDump)
        turnOnCoreDumps();

    hiveGlobalRankId = 0;
    hiveGlobalRankCount = config->tableLength;
    hiveServerSetup(config);
    hiveGlobalMasterRankId= config->masterRank;
    if(hiveGlobalRankId == config->masterRank && config->masterBoot)
        config->launcherData->launchProcesses(config->launcherData);

    if(hiveGlobalRankCount>1)
    {
        hiveRemoteSetupOutgoing();
        if(!hiveRemoteSetupIncoming())
            return -1;
    }

    hiveGuidTableInit(config->routeTableSize);
    hiveThreadInit(config);
    hiveThreadZeroNodeStart();

    hiveThreadMainJoin();
    hiveRemoteCleanup();

    if(hiveGlobalRankId == config->masterRank && config->masterBoot)
    {
        config->launcherData->cleanupProcesses(config->launcherData);
    }
    hiveConfigDestroy(config);
    hiveRemoteTryToClosePrinter();
    return 0;
}
