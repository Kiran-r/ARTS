#ifndef HIVEDBLIST_H
#define HIVEDBLIST_H

#define DBSPERELEMENT 8

struct hiveDbElement
{
    struct hiveDbElement * next;
    unsigned int array[DBSPERELEMENT];
};

struct hiveLocalDelayedEdt
{
    struct hiveLocalDelayedEdt * next;
    struct hiveEdt * edt[DBSPERELEMENT];
    unsigned int slot[DBSPERELEMENT];
    hiveDbAccessMode_t mode[DBSPERELEMENT];
};

struct hiveDbFrontier
{
    struct hiveDbElement list;
    unsigned int position;
    struct hiveDbFrontier * next;
    volatile unsigned int lock;
    
    /* 
     * This is because we can't aggregate exclusive requests
     * and we need to store them somewhere.  There will only
     * be at most one per frontier.
     */
    unsigned int exNode;
    struct hiveEdt * exEdt;
    unsigned int exSlot;
    hiveDbAccessMode_t exMode;
    
    /* 
     * This is dumb, but we need somewhere to store requests
     * that are from the guid owner but cannot be satisfied
     * because of the memory model
     */
    unsigned int localPosition;
    struct hiveLocalDelayedEdt localDelayed;
};

struct hiveDbList
{
    struct hiveDbFrontier * head;
    struct hiveDbFrontier * tail;
    volatile unsigned int reader;
    volatile unsigned int writer;
};

struct hiveDbFrontierIterator
{
    struct hiveDbFrontier * frontier;
    unsigned int currentIndex;
    struct hiveDbElement * currentElement;
};

struct hiveDbList * hiveNewDbList();
unsigned int hiveCurrentFrontierSize(struct hiveDbList * dbList);
struct hiveDbFrontierIterator * hiveDbFrontierIterCreate(struct hiveDbFrontier * frontier);
unsigned int hiveDbFrontierIterSize(struct hiveDbFrontierIterator * iter);
bool hiveDbFrontierIterNext(struct hiveDbFrontierIterator * iter, unsigned int * next);
bool hiveDbFrontierIterHasNext(struct hiveDbFrontierIterator * iter);
void hiveDbFrontierIterDelete(struct hiveDbFrontierIterator * iter);
void hiveProgressFrontier(struct hiveDb * db, unsigned int rank);
struct hiveDbFrontierIterator * hiveProgressAndGetFrontier(struct hiveDbList * dbList);
bool hivePushDbToList(struct hiveDbList * dbList, unsigned int data, bool write, bool exclusive, bool local, bool bypass, struct hiveEdt * edt, unsigned int slot, hiveDbAccessMode_t mode);
struct hiveDbFrontierIterator * hiveCloseFrontier(struct hiveDbList * dbList);

#endif /* HIVEDBLIST_H */

