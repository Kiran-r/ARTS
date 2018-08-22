#ifndef ARTSMALLOC_H
#define ARTSMALLOC_H
#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"

void *artsMalloc(size_t size);
void *artsMallocAlign(size_t size, size_t align);
void *artsCalloc(size_t size);
void *artsCallocAlign(size_t size, size_t allign);
void * artsRealloc(void *ptr, size_t size);
void artsFree(void *ptr);
void artsFreeAlign(void *ptr);
#ifdef __cplusplus
}
#endif

#endif
