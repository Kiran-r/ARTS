#ifndef HIVETHREADS_H
#define HIVETHREADS_H

#include "hiveConfig.h"

void
hiveThreadInit( struct hiveConfig * config  );
void
hiveThreadMainJoin();
void hiveThreadPin(void * data);
void hiveThreadSetOsThreadCount(unsigned int threads);

#endif
