#ifndef ARTSEVENTFUNCTIONS_H
#define ARTSEVENTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#define INITIAL_DEPENDENT_SIZE 4
    
bool artsEventCreateInternal( artsGuid_t * guid, 
                              unsigned int route, 
                              unsigned int dependentCount, 
                              unsigned int latchCount, 
                              bool destroyOnFire );

artsGuid_t artsEventCreateLatch(unsigned int route, unsigned int latchCount);

artsGuid_t artsEventCreateLatchWithGuid(artsGuid_t guid, unsigned int latchCount);

void artsEventDestroy(artsGuid_t guid);

void artsEventSatisfySlot(artsGuid_t eventGuid, artsGuid_t dataGuid, uint32_t slot);

void artsAddDependence(artsGuid_t source, artsGuid_t destination, uint32_t slot);

void artsAddLocalEventCallback(artsGuid_t source, eventCallback_t callback);

bool artsIsEventFiredExt(artsGuid_t event);

#ifdef __cplusplus
}
#endif

#endif /* ARTSEVENTFUNCTIONS_H */

