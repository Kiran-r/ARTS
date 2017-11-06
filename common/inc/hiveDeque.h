#ifndef HIVEDEQUE_H
#define HIVEDEQUE_H

#include "hive.h"

struct hiveDeque;
struct hiveDeque *hiveDequeListNew(unsigned int listSize, unsigned int dequeSize);
struct hiveDeque *hiveDequeListGetDeque(struct hiveDeque *dequeList, unsigned int position);
void hiveDequeListDelete(void *dequeList);
struct hiveDeque * hiveDequeNew(unsigned int size);
void hiveDequeDelete(struct hiveDeque *deque);
bool hiveDequePushFront(struct hiveDeque *deque, void *item, unsigned int priority);
void *hiveDequePopFront(struct hiveDeque *deque);
void *hiveDequePopBack(struct hiveDeque *deque);
unsigned int hiveDequePopBackMultiple(struct hiveDeque *deque, unsigned int amount, void ** returnList );
bool hiveDequeEmpty(struct hiveDeque *deque);
void hiveDequeClear(struct hiveDeque *deque);
unsigned int hiveDequeSize(struct hiveDeque *deque);

#endif
