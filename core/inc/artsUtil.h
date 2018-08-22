#ifndef ARTSEXTENSIONS_H
#define ARTSEXTENSIONS_H
#ifdef __cplusplus
extern "C" {
#endif
artsGuid_t artsGetCurrentGuid();
unsigned int artsGetCurrentNode();
unsigned int artsGetTotalNodes();
unsigned int artsGetCurrentWorker();
unsigned int artsGetTotalWorkers();
unsigned int artsGetCurrentCluster();
unsigned int artsGetTotalClusters();

#define NAMEDGUIDS(...) enum __namedGuidIndex{ firstNamedGuid=0, __VA_ARGS__, lastNamedGuid }; artsGuid_t __namedGuid[lastNamedGuid] = { 0 }
#define GETGUID(name) __namedGuid[name]
#define SETGUID(name, guid) __namedGuid[name] = guid

#define NAMEDGUIDARRAYS(...) enum __namedArrayGuidIndex{ firstNamedArrayGuid=0, __VA_ARGS__, lastNamedArrayGuid }; artsGuid_t * __namedArrayGuid[lastNamedArrayGuid] = { 0 }
#define GETGUIDARRAYOFFSET(name, offset) __namedArrayGuid[name][offset]
#define GETGUIDARRAY(name) __namedArrayGuid[name]
#define SETGUIDARRAY(name, guidPtr) __namedArrayGuid[name] = guidPtr
#ifdef __cplusplus
}
#endif
#endif

