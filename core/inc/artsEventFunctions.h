#ifndef ARTSEVENTFUNCTIONS_H
#define ARTSEVENTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#define INITIAL_DEPENDENT_SIZE 4
    
bool artsEventCreateInternal( artsGuid_t * guid, 
                              unsigned int route, 
                              unsigned int dependentCount, 
                              unsigned int latchCount, 
                              bool destroyOnFire );



#ifdef __cplusplus
}
#endif

#endif /* ARTSEVENTFUNCTIONS_H */

