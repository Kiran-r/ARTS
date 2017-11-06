#ifndef HIVEOUTOFORDERLIST_H
#define HIVEOUTOFORDERLIST_H

#include "hive.h"
#define OOPERELEMENT 4

struct hiveOutOfOrderElement
{
    volatile struct hiveOutOfOrderElement * next;
    volatile void * array[OOPERELEMENT];
};

struct hiveOutOfOrderList
{
    volatile unsigned int readerLock;
    volatile unsigned int writerLock;
    volatile unsigned int count;
    bool isFired;
    struct hiveOutOfOrderElement head;
};

bool hiveOutOfOrderListAddItem(struct hiveOutOfOrderList * addToMe, void * item);
void hiveOutOfOrderListFireCallback(struct hiveOutOfOrderList* fireMe, void * localGuidAddress,  void (* callback)(void *, void *));
void hiveOutOfOrderListReset(struct hiveOutOfOrderList* fireMe);
void hiveOutOfOrderListDelete(struct hiveOutOfOrderList* fireMe);

#endif
