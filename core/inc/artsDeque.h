#ifndef ARTSDEQUE_H
#define ARTSDEQUE_H

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
unsigned int artsDequePopBackMultiple(struct artsDeque *deque, unsigned int amount, void ** returnList );
void ** artsDequePopBackMult(struct artsDeque *deque);
bool artsDequeEmpty(struct artsDeque *deque);
void artsDequeClear(struct artsDeque *deque);
unsigned int artsDequeSize(struct artsDeque *deque);

#endif
