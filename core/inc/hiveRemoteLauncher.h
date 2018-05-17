#ifndef HIVEREMOTELAUNCHER_H
#define HIVEREMOTELAUNCHER_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hiveConfig.h"      //For struct hiveConfig
#include "hiveMalloc.h"       //For hiveMalloc/hiveFree

struct hiveRemoteLauncher
{
    unsigned int argc;
    char ** argv;
    struct hiveConfig * config;
    unsigned int killStuckProcesses;
    void (*launchProcesses)( struct hiveRemoteLauncher * );  
    void (*cleanupProcesses)( struct hiveRemoteLauncher * );  
    void * launcherMemory;
};

//Add your launcher prototypes here
void hiveRemoteLauncherSSHStartupProcesses( struct hiveRemoteLauncher * launcher );
void hiveRemoteLauncherSSHCleanupProcesses( struct hiveRemoteLauncher * launcher );

static inline struct hiveRemoteLauncher * hiveRemoteLauncherCreate( unsigned int argc, char ** argv, struct hiveConfig * config, unsigned int killMode, void (*launchProcesses)( struct hiveRemoteLauncher * ),  void (*cleanupProcesses)( struct hiveRemoteLauncher * ) )  
{
    struct hiveRemoteLauncher * launcher = hiveMalloc( sizeof(struct hiveRemoteLauncher) );

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
