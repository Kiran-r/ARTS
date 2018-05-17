#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hiveMalloc.h"
#include "hiveArrayList.h"
#include "stdint.h"
#include "inttypes.h"

hiveArrayListElement * hiveNewArrayListElement(uint64_t start, size_t elementSize, size_t arrayLength)
{
    hiveArrayListElement * ret = (hiveArrayListElement*) hiveMalloc(sizeof(hiveArrayListElement) + elementSize * arrayLength);  
    ret->start = start;
    ret->next = NULL;
    ret->array = (void*)(1+ret);
    return ret;
}

hiveArrayList * hiveNewArrayList(size_t elementSize, size_t arrayLength)
{
    hiveArrayList * ret = (hiveArrayList*) hiveMalloc(sizeof(hiveArrayList));
    ret->elementSize = elementSize;
    ret->arrayLength = arrayLength;
    ret->head = ret->current = hiveNewArrayListElement(0, elementSize, arrayLength);
    ret->index = 0;
    ret->lastRequest = 0;
    ret->lastRequestPtr = ret->head->array;
    return ret;
}

void hiveDeleteArrayList(hiveArrayList * aList)
{
    hiveArrayListElement * trail;
    hiveArrayListElement * current = aList->head;
    while(current)
    {
        trail = current;
        current = current->next;
        hiveFree(trail);
    }
    hiveFree(aList);    
}

uint64_t hivePushToArrayList(hiveArrayList * aList, void * element)
{
    uint64_t index = aList->index;
    if(!(aList->index % aList->arrayLength) && aList->index)
    {
        if(!aList->current->next)
            aList->current->next = hiveNewArrayListElement(aList->current->start+aList->arrayLength, aList->elementSize, aList->arrayLength);
        aList->current = aList->current->next;
    }
    uint64_t offset =  aList->index - aList->current->start;
    void * ptr = (void*)((char*)aList->current->array + offset*aList->elementSize);
    memcpy(ptr, element, aList->elementSize);
    aList->index++;
    return index;
}

void hiveResetArrayList(hiveArrayList * aList)
{
    aList->current = aList->head;
    aList->index = 0;
    aList->lastRequest = 0;
    aList->lastRequestPtr = aList->head->array;
}

uint64_t hiveLengthArrayList(hiveArrayList * aList)
{
    return aList->index;
}

void * hiveGetFromArrayList(hiveArrayList * aList, uint64_t index)
{
    if(aList)
    {
        //Fastest Path
        if(index==aList->lastRequest)
            return aList->lastRequestPtr;

        if(index < aList->index)
        {           
            aList->lastRequest = index;

            //Faster Path
            if(aList->index < aList->arrayLength)
            {
                aList->lastRequestPtr = (void*) ((char*)aList->head->array + index * aList->elementSize);
                return aList->lastRequestPtr;
            }

            //Slow Path
            hiveArrayListElement * node = aList->head;
            while(node && index >= node->start + aList->arrayLength )
                node = node->next;
            if(node)
            {
                uint64_t offset =  index - node->start;
                aList->lastRequestPtr = (void*) ((char*)node->array + offset * aList->elementSize);
                return aList->lastRequestPtr; 
            }
        }
    }
    return NULL;
}

    hiveArrayListIterator * hiveNewArrayListIterator(hiveArrayList * aList)
    {
        hiveArrayListIterator * iter = hiveMalloc(sizeof(hiveArrayListIterator));
        iter->index = 0;
        iter->last = aList->index;
        iter->elementSize = aList->elementSize;
        iter->arrayLength = aList->arrayLength;
        iter->current = aList->head;

        return iter;
    }
    
    void * hiveArrayListNext(hiveArrayListIterator * iter)
    {
        void * ret = NULL;
        if(iter)
        {
            if(iter->index < iter->last)
            {
                if(!(iter->index % iter->arrayLength) && iter->index)
                {
                    iter->current = iter->current->next;
                }
                if(iter->current)
                {
                    ret = (void*) ((char*)iter->current->array + (iter->index - iter->current->start) * iter->elementSize);
                    iter->index++;
                }
            }
        }
        return ret;
    }
    
    bool hiveArrayListHasNext(hiveArrayListIterator * iter)
    {
        return (iter->index < iter->last);
    }
    
    void hiveDeleteArrayListIterator(hiveArrayListIterator * iter)
    {
        hiveFree(iter);
    }
