#ifndef ARTSDEQUE_H
#define ARTSDEQUE_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"  
#define STEALSIZE 1024

struct artsDeque;
struct artsDeque *artsDequeListNew(unsigned int listSize, unsigned int dequeSize);
struct artsDeque *artsDequeListGetDeque(struct artsDeque *dequeList, unsigned int position);
void artsDequeListDelete(void *dequeList);
struct artsDeque * artsDequeNew(unsigned int size);
void artsDequeDelete(struct artsDeque *deque);
bool artsDequePushFront(struct artsDeque *deque, void *item, unsigned int priority);
void *artsDequePopFront(struct artsDeque *deque);
void *artsDequePopBack(struct artsDeque *deque);

bool artsDequeEmpty(struct artsDeque *deque);
void artsDequeClear(struct artsDeque *deque);
unsigned int artsDequeSize(struct artsDeque *deque);

#ifdef __cplusplus
}
#endif
#endif
