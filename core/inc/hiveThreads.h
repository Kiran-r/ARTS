#ifndef HIVETHREADS_H
#define HIVETHREADS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hiveConfig.h"

void hiveThreadInit(struct hiveConfig * config);
void hiveThreadMainJoin();
void hiveThreadSetOsThreadCount(unsigned int threads);
#ifdef __cplusplus
}
#endif
#endif
