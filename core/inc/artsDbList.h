#ifndef ARTSDBLIST_H
#define ARTSDBLIST_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
    
#define DBSPERELEMENT 8

struct artsDbElement
{
    struct artsDbElement * next;
    unsigned int array[DBSPERELEMENT];
};

struct artsLocalDelayedEdt
{
    struct artsLocalDelayedEdt * next;
    struct artsEdt * edt[DBSPERELEMENT];
    unsigned int slot[DBSPERELEMENT];
    artsType_t mode[DBSPERELEMENT];
};

struct artsDbFrontier
{
    struct artsDbElement list;
    unsigned int position;
    struct artsDbFrontier * next;
    volatile unsigned int lock;
    
    /* 
     * This is because we can't aggregate exclusive requests
     * and we need to store them somewhere.  There will only
     * be at most one per frontier.
     */
    unsigned int exNode;
    struct artsEdt * exEdt;
    unsigned int exSlot;
    artsType_t exMode;
    
    /* 
     * This is dumb, but we need somewhere to store requests
     * that are from the guid owner but cannot be satisfied
     * because of the memory model
     */
    unsigned int localPosition;
    struct artsLocalDelayedEdt localDelayed;
};

struct artsDbList
{
    struct artsDbFrontier * head;
    struct artsDbFrontier * tail;
    volatile unsigned int reader;
    volatile unsigned int writer;
};

struct artsDbFrontierIterator
{
    struct artsDbFrontier * frontier;
    unsigned int currentIndex;
    struct artsDbElement * currentElement;
};

struct artsDbList * artsNewDbList();
unsigned int artsCurrentFrontierSize(struct artsDbList * dbList);
struct artsDbFrontierIterator * artsDbFrontierIterCreate(struct artsDbFrontier * frontier);
unsigned int artsDbFrontierIterSize(struct artsDbFrontierIterator * iter);
bool artsDbFrontierIterNext(struct artsDbFrontierIterator * iter, unsigned int * next);
bool artsDbFrontierIterHasNext(struct artsDbFrontierIterator * iter);
void artsDbFrontierIterDelete(struct artsDbFrontierIterator * iter);
void artsProgressFrontier(struct artsDb * db, unsigned int rank);
struct artsDbFrontierIterator * artsProgressAndGetFrontier(struct artsDbList * dbList);
bool artsPushDbToList(struct artsDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct artsEdt * edt, unsigned int slot, artsType_t mode);
struct artsDbFrontierIterator * artsCloseFrontier(struct artsDbList * dbList);
#ifdef __cplusplus
}
#endif

#endif /* ARTSDBLIST_H */

