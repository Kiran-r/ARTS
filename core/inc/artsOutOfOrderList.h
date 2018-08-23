#ifndef ARTSOUTOFORDERLIST_H
#define ARTSOUTOFORDERLIST_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"
#define OOPERELEMENT 4

struct artsOutOfOrderElement
{
    volatile struct artsOutOfOrderElement * next;
    volatile void * array[OOPERELEMENT];
};

struct artsOutOfOrderList
{
    volatile unsigned int readerLock;
    volatile unsigned int writerLock;
    volatile unsigned int count;
    bool isFired;
    struct artsOutOfOrderElement head;
};

bool artsOutOfOrderListAddItem(struct artsOutOfOrderList * addToMe, void * item);
void artsOutOfOrderListFireCallback(struct artsOutOfOrderList* fireMe, void * localGuidAddress,  void (* callback)(void *, void *));
void artsOutOfOrderListReset(struct artsOutOfOrderList* fireMe);
void artsOutOfOrderListDelete(struct artsOutOfOrderList* fireMe);
#ifdef __cplusplus
}
#endif

#endif
