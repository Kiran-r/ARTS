#ifndef HIVEABSTRACTMACHINEMODEL_H
#define	HIVEABSTRACTMACHINEMODEL_H

#include "hiveMalloc.h"
#include "hiveConfig.h"
    struct hiveCoreInfo;
    struct unitThread
    {
        unsigned int id;
        unsigned int groupId;
        unsigned int groupPos;
        bool worker;
        bool networkSend;
        bool networkReceive;
        bool acdt;
        bool statusSend;
        bool sim;
        bool pin;
        struct unitThread * next;
    };
    struct threadMask
    {
        unsigned int clusterId;
        unsigned int coreId;
        unsigned int unitId;
        bool on;
        //unsigned int threads;
        unsigned int id;
        unsigned int groupId;
        unsigned int groupPos;
        bool worker;
        bool networkSend;
        bool networkReceive;
        bool acdt;
        bool statusSend;
        bool sim;
        bool pin;
        struct hiveCoreInfo * coreInfo;
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
        struct hiveCoreInfo * coreInfo;
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
    
struct threadMask * getThreadMask(struct hiveConfig * config);
void printMask(struct threadMask * units, unsigned int numberOfUnits);
void hiveAbstractMachineModelPinThread(struct hiveCoreInfo * coreInfo );

#endif	/* hiveABSTRACTMACHINEMODEL_H */

