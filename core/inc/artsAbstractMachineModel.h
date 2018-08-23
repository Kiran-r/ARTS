#ifndef ARTSABSTRACTMACHINEMODEL_H
#define	ARTSABSTRACTMACHINEMODEL_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "artsConfig.h"
    
struct artsCoreInfo;
struct unitThread
{
    unsigned int id;
    unsigned int groupId;
    unsigned int groupPos;
    bool worker;
    bool networkSend;
    bool networkReceive;
    bool statusSend;
    bool pin;
    struct unitThread * next;
};

struct threadMask
{
    unsigned int clusterId;
    unsigned int coreId;
    unsigned int unitId;
    bool on;
    unsigned int id;
    unsigned int groupId;
    unsigned int groupPos;
    bool worker;
    bool networkSend;
    bool networkReceive;
    bool statusSend;
    bool pin;
    struct artsCoreInfo * coreInfo;
};

struct unitMask
{
    unsigned int clusterId;
    unsigned int coreId;
    unsigned int unitId;
    bool on;
    unsigned int threads;
    struct unitThread * listHead;
    struct unitThread * listTail;
    struct artsCoreInfo * coreInfo;
};

struct coreMask
{
    unsigned int numUnits;
    struct unitMask * unit;
};

struct clusterMask 
{
    unsigned int numCores;
    struct coreMask * core;
};

struct nodeMask
{
    unsigned int numClusters;
    struct clusterMask * cluster;
};
    
struct threadMask * getThreadMask(struct artsConfig * config);
void printMask(struct threadMask * units, unsigned int numberOfUnits);
void artsAbstractMachineModelPinThread(struct artsCoreInfo * coreInfo);

#ifdef __cplusplus
}
#endif

#endif	/* artsABSTRACTMACHINEMODEL_H */

