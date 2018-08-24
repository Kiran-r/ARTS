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
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include "arts.h"
#include "artsThreads.h"
#include "artsConfig.h"
#include "artsGlobals.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "artsRuntime.h"
#include "artsRemoteLauncher.h"
#include "artsIntrospection.h"
#include "artsDebug.h"

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
