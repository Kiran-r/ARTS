#ifndef ARTSARRAYLIST_H
#define	ARTSARRAYLIST_H

#include "arts.h"

    typedef struct artsArrayListElement artsArrayListElement;

    struct artsArrayListElement {
        uint64_t start;
        artsArrayListElement * next;
        void * array;
    };

    typedef struct {
        size_t elementSize;
        size_t arrayLength;
        artsArrayListElement * head;
        artsArrayListElement * current;
        uint64_t index;
        uint64_t lastRequest;
        void * lastRequestPtr;
    } artsArrayList;

    typedef struct {
        uint64_t index;
        uint64_t last;
        size_t elementSize;
        size_t arrayLength;
        artsArrayListElement * current;
    } artsArrayListIterator;
    
    artsArrayListElement * artsNewArrayListElement(uint64_t start, size_t elementSize, size_t arrayLength);
    artsArrayList * artsNewArrayList(size_t elementSize, size_t arrayLength);
    void artsDeleteArrayList(artsArrayList * aList);
    uint64_t artsPushToArrayList(artsArrayList * aList, void * element);
    void artsResetArrayList(artsArrayList * aList);
    uint64_t artsLengthArrayList(artsArrayList * aList);
    void * artsGetFromArrayList(artsArrayList * aList, uint64_t index);
    artsArrayListIterator * artsNewArrayListIterator(artsArrayList * aList);
    void * artsArrayListNext(artsArrayListIterator * iter);
    bool artsArrayListHasNext(artsArrayListIterator * iter);
    void artsDeleteArrayListIterator(artsArrayListIterator * iter);

#endif	/* LIST_H */

