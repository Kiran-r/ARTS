#ifndef ARTSREMOTELAUNCHER_H
#define ARTSREMOTELAUNCHER_H
#ifdef __cplusplus
extern "C" {
#endif
#include "artsConfig.h"      //For struct artsConfig
#include "artsMalloc.h"       //For artsMalloc/artsFree

struct artsRemoteLauncher
{
    unsigned int argc;
    char ** argv;
    struct artsConfig * config;
    unsigned int killStuckProcesses;
    void (*launchProcesses)( struct artsRemoteLauncher * );  
    void (*cleanupProcesses)( struct artsRemoteLauncher * );  
    void * launcherMemory;
};

//Add your launcher prototypes here
void artsRemoteLauncherSSHStartupProcesses( struct artsRemoteLauncher * launcher );
void artsRemoteLauncherSSHCleanupProcesses( struct artsRemoteLauncher * launcher );

static inline struct artsRemoteLauncher * artsRemoteLauncherCreate( unsigned int argc, char ** argv, struct artsConfig * config, unsigned int killMode, void (*launchProcesses)( struct artsRemoteLauncher * ),  void (*cleanupProcesses)( struct artsRemoteLauncher * ) )  
{
    struct artsRemoteLauncher * launcher = artsMalloc( sizeof(struct artsRemoteLauncher) );

    launcher->argc = argc;
    launcher->argv = argv;
    launcher->config = config;
    launcher->killStuckProcesses = killMode;
    launcher->launcherMemory = NULL;
    launcher->launchProcesses = launchProcesses;
    launcher->cleanupProcesses = cleanupProcesses;

    return launcher;
}
#ifdef __cplusplus
}
#endif

#endif
