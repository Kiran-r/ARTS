#ifndef HIVEEXTENSIONS_H
#define HIVEEXTENSIONS_H
#ifdef __cplusplus
extern "C" {
#endif
hiveGuid_t hiveGetCurrentGuid();
unsigned int hiveGetCurrentNode();
unsigned int hiveGetTotalNodes();
unsigned int hiveGetCurrentWorker();
unsigned int hiveGetTotalWorkers();
unsigned int hiveGetCurrentCluster();
unsigned int hiveGetTotalClusters();

#define NAMEDGUIDS(...) enum __namedGuidIndex{ firstNamedGuid=0, __VA_ARGS__, lastNamedGuid }; hiveGuid_t __namedGuid[lastNamedGuid] = { 0 }
#define GETGUID(name) __namedGuid[name]
#define SETGUID(name, guid) __namedGuid[name] = guid

#define NAMEDGUIDARRAYS(...) enum __namedArrayGuidIndex{ firstNamedArrayGuid=0, __VA_ARGS__, lastNamedArrayGuid }; hiveGuid_t * __namedArrayGuid[lastNamedArrayGuid] = { 0 }
#define GETGUIDARRAYOFFSET(name, offset) __namedArrayGuid[name][offset]
#define GETGUIDARRAY(name) __namedArrayGuid[name]
#define SETGUIDARRAY(name, guidPtr) __namedArrayGuid[name] = guidPtr
#ifdef __cplusplus
}
#endif
#endif

