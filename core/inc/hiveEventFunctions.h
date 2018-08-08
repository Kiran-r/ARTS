#ifndef HIVEEVENTFUNCTIONS_H
#define HIVEEVENTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#define INITIAL_DEPENDENT_SIZE 4
    
bool hiveEventCreateInternal( hiveGuid_t * guid, 
                              unsigned int route, 
                              unsigned int dependentCount, 
                              unsigned int latchCount, 
                              bool destroyOnFire );

hiveGuid_t hiveEventCreateLatch(unsigned int route, unsigned int latchCount);

hiveGuid_t hiveEventCreateLatchWithGuid(hiveGuid_t guid, unsigned int latchCount);

void hiveEventDestroy(hiveGuid_t guid);

void hiveEventSatisfySlot(hiveGuid_t eventGuid, hiveGuid_t dataGuid, u32 slot);

void hiveAddDependence(hiveGuid_t source, hiveGuid_t destination, u32 slot, hiveDbAccessMode_t mode);

void hiveAddLocalEventCallback(hiveGuid_t source, eventCallback_t callback);

bool hiveIsEventFiredExt(hiveGuid_t event);

#ifdef __cplusplus
}
#endif

#endif /* HIVEEVENTFUNCTIONS_H */

