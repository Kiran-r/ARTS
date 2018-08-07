#ifndef HIVECONFIG_H
#define HIVECONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include "hive.h"
#include "hiveRemoteLauncher.h"

struct hiveConfigTable
{
    unsigned int rank;
    char * ipAddress;
};

struct hiveConfigVariable
{
    unsigned int size;
    struct hiveConfigVariable * next;
    char variable[255];
    char value[];
};

struct hiveConfig
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
    u64 stackSize;
    struct hiveRemoteLauncher * launcherData;
    char * introspectiveFolder;
    char * introspectiveConf;
    char * counterFolder;
    unsigned int tableLength;
    struct hiveConfigTable * table;
};

struct hiveConfig * hiveConfigLoad( int argc, char ** argv, char * location );
void hiveConfigDestroy( void * config );
unsigned int hiveConfigGetNumberOfThreads(char * location);
#ifdef __cplusplus
}
#endif

#endif
