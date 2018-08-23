#ifndef ARTSTHREADS_H
#define ARTSTHREADS_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "artsConfig.h"

void artsThreadInit(struct artsConfig * config);
void artsThreadMainJoin();
void artsThreadSetOsThreadCount(unsigned int threads);
#ifdef __cplusplus
}
#endif
#endif
