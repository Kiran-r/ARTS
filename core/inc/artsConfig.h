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
    bool remoteWorkStealing;
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
    struct artsConfigTable * table;
};

struct artsConfig * artsConfigLoad( int argc, char ** argv, char * location );
void artsConfigDestroy( void * config );
unsigned int artsConfigGetNumberOfThreads(char * location);
#ifdef __cplusplus
}
#endif

#endif
