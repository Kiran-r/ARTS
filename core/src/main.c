#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include "arts.h"
#include "artsThreads.h"
#include "artsConfig.h"
#include "artsGlobals.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "artsRuntime.h"
#include "artsMalloc.h"
#include "artsRemoteLauncher.h"
#include "artsIntrospection.h"
#include "artsDebug.h"
#include <string.h>

extern struct artsConfig * config;

int mainArgc = 0;
char ** mainArgv = NULL;

int artsRT(int argc, char **argv)
{
    mainArgc = argc;
    mainArgv = argv;
    artsRemoteTryToBecomePrinter();
    config = artsConfigLoad(0, NULL, NULL);

    if(config->coreDump)
        artsTurnOnCoreDumps();

    artsGlobalRankId = 0;
    artsGlobalRankCount = config->tableLength;
    if(strncmp(config->launcher, "local", 5) != 0)
        artsServerSetup(config);
    artsGlobalMasterRankId= config->masterRank;
    if(artsGlobalRankId == config->masterRank && config->masterBoot)
        config->launcherData->launchProcesses(config->launcherData);

    if(artsGlobalRankCount>1)
    {
        artsRemoteSetupOutgoing();
        if(!artsRemoteSetupIncoming())
            return -1;
    }

    artsThreadInit(config);
    artsThreadZeroNodeStart();
       
    artsThreadMainJoin();
    artsRemoteCleanup();

    if(artsGlobalRankId == config->masterRank && config->masterBoot)
    {
        config->launcherData->cleanupProcesses(config->launcherData);
    }
    artsConfigDestroy(config);
    artsRemoteTryToClosePrinter();
    return 0;
}
