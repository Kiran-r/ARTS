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
#ifndef ARTSCONFIG_H
#define ARTSCONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "artsRemoteLauncher.h"

struct artsConfigTable
{
    unsigned int rank;
    char * ipAddress;
};

struct artsConfigVariable
{
    unsigned int size;
    struct artsConfigVariable * next;
    char variable[255];
    char value[];
};

struct artsConfig
{
    unsigned int myRank;
    char * masterNode;
    char * myIPAddress;
    char * netInterface;
    char * protocol;
    char * launcher;
    unsigned int ports;
    unsigned int osThreadCount;
    unsigned int threadCount;
    unsigned int coreCount;
    unsigned int recieverCount;
    unsigned int senderCount;
    unsigned int socketCount;
    unsigned int nodes;
    unsigned int masterRank;
    unsigned int port;
    unsigned int killMode;
    unsigned int routeTableSize;
    unsigned int routeTableEntries;
    unsigned int dequeSize;
    unsigned int introspectiveTraceLevel;
    unsigned int introspectiveStartPoint;
    unsigned int counterStartPoint;
    unsigned int printNodeStats;
    unsigned int scheduler;
    unsigned int shutdownEpoch;
    char * prefix;
    char * suffix;
    bool ibNames;
    bool masterBoot;
    bool coreDump;
    unsigned int pinStride;
    bool printTopology;
    bool pinThreads;
    unsigned int firstEdt;
    unsigned int shadLoopStride;
    uint64_t stackSize;
    struct artsRemoteLauncher * launcherData;
    char * introspectiveFolder;
    char * introspectiveConf;
    char * counterFolder;
    unsigned int tableLength;
    unsigned int tMT;  // @awmm temporal MT; # of MT aliases per core thread; 0 if disabled
    struct artsConfigTable * table;
};

struct artsConfig * artsConfigLoad( int argc, char ** argv, char * location );
void artsConfigDestroy( void * config );
unsigned int artsConfigGetNumberOfThreads(char * location);
#ifdef __cplusplus
}
#endif

#endif
