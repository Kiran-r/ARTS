//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#ifndef ARTSREMOTELAUNCHER_H
#define ARTSREMOTELAUNCHER_H
#ifdef __cplusplus
extern "C" {
#endif
#include "artsConfig.h"      //For struct artsConfig

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
