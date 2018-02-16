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
#include "hiveDebug.h"

extern struct hiveConfig * config;

int mainArgc = 0;
char ** mainArgv = NULL;

int hiveRT(int argc, char **argv)
{
    mainArgc = argc;
    mainArgv = argv;
    hiveRemoteTryToBecomePrinter();
    config = hiveConfigLoad(0, NULL, NULL);

    if(config->coreDump)
        hiveTurnOnCoreDumps();

    hiveGlobalRankId = 0;
    hiveGlobalRankCount = config->tableLength;
    if(strncmp(config->launcher, "local", 5) != 0) 
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
