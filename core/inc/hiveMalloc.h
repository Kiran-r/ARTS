#ifndef HIVEMALLOC_H
#define HIVEMALLOC_H

#include "stddef.h"

void *hiveMalloc(size_t size);
void *hiveMallocAlign(size_t size, size_t align);
void *hiveCalloc(size_t size);
void *hiveCallocAlign(size_t size, size_t allign);
void * hiveRealloc(void *ptr, size_t size);
void hiveFree(void *ptr);
void hiveFreeAlign(void *ptr);

#endif
