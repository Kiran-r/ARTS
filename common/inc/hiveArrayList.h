#ifndef HIVEARRAYLIST_H
#define	HIVEARRAYLIST_H

#include "hive.h"

    typedef struct hiveArrayListElement hiveArrayListElement;

    struct hiveArrayListElement {
        uint64_t start;
        hiveArrayListElement * next;
        void * array;
    };

    typedef struct {
        size_t elementSize;
        size_t arrayLength;
        hiveArrayListElement * head;
        hiveArrayListElement * current;
        uint64_t index;
        uint64_t lastRequest;
        void * lastRequestPtr;
    } hiveArrayList;

    typedef struct {
        uint64_t index;
        uint64_t last;
        size_t elementSize;
        size_t arrayLength;
        hiveArrayListElement * current;
    } hiveArrayListIterator;
    
    hiveArrayListElement * hiveNewArrayListElement(uint64_t start, size_t elementSize, size_t arrayLength);
    hiveArrayList * hiveNewArrayList(size_t elementSize, size_t arrayLength);
    void hiveDeleteArrayList(hiveArrayList * aList);
    uint64_t hivePushToArrayList(hiveArrayList * aList, void * element);
    void hiveResetArrayList(hiveArrayList * aList);
    uint64_t hiveLengthArrayList(hiveArrayList * aList);
    void * hiveGetFromArrayList(hiveArrayList * aList, uint64_t index);
    hiveArrayListIterator * hiveNewArrayListIterator(hiveArrayList * aList);
    void * hiveArrayListNext(hiveArrayListIterator * iter);
    bool hiveArrayListHasNext(hiveArrayListIterator * iter);
    void hiveDeleteArrayListIterator(hiveArrayListIterator * iter);

#endif	/* LIST_H */

