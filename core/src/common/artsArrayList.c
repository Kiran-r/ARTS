//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#include "artsArrayList.h"

artsArrayListElement * artsNewArrayListElement(uint64_t start, size_t elementSize, size_t arrayLength)
{
    artsArrayListElement * ret = (artsArrayListElement*) artsMalloc(sizeof(artsArrayListElement) + elementSize * arrayLength);  
    ret->start = start;
    ret->next = NULL;
    ret->array = (void*)(1+ret);
    return ret;
}

artsArrayList * artsNewArrayList(size_t elementSize, size_t arrayLength)
{
    artsArrayList * ret = (artsArrayList*) artsMalloc(sizeof(artsArrayList));
    ret->elementSize = elementSize;
    ret->arrayLength = arrayLength;
    ret->head = ret->current = artsNewArrayListElement(0, elementSize, arrayLength);
    ret->index = 0;
    ret->lastRequest = 0;
    ret->lastRequestPtr = ret->head->array;
    return ret;
}

void artsDeleteArrayList(artsArrayList * aList)
{
    artsArrayListElement * trail;
    artsArrayListElement * current = aList->head;
    while(current)
    {
        trail = current;
        current = current->next;
        artsFree(trail);
    }
    artsFree(aList);    
}

uint64_t artsPushToArrayList(artsArrayList * aList, void * element)
{
    uint64_t index = aList->index;
    if(!(aList->index % aList->arrayLength) && aList->index)
    {
        if(!aList->current->next)
            aList->current->next = artsNewArrayListElement(aList->current->start+aList->arrayLength, aList->elementSize, aList->arrayLength);
        aList->current = aList->current->next;
    }
    uint64_t offset =  aList->index - aList->current->start;
    void * ptr = (void*)((char*)aList->current->array + offset*aList->elementSize);
    memcpy(ptr, element, aList->elementSize);
    aList->index++;
    return index;
}

void artsResetArrayList(artsArrayList * aList)
{
    aList->current = aList->head;
    aList->index = 0;
    aList->lastRequest = 0;
    aList->lastRequestPtr = aList->head->array;
}

uint64_t artsLengthArrayList(artsArrayList * aList)
{
    return aList->index;
}

void * artsGetFromArrayList(artsArrayList * aList, uint64_t index)
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
            artsArrayListElement * node = aList->head;
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

    artsArrayListIterator * artsNewArrayListIterator(artsArrayList * aList)
    {
        artsArrayListIterator * iter = artsMalloc(sizeof(artsArrayListIterator));
        iter->index = 0;
        iter->last = aList->index;
        iter->elementSize = aList->elementSize;
        iter->arrayLength = aList->arrayLength;
        iter->current = aList->head;

        return iter;
    }
    
    void * artsArrayListNext(artsArrayListIterator * iter)
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
    
    bool artsArrayListHasNext(artsArrayListIterator * iter)
    {
        return (iter->index < iter->last);
    }
    
    void artsDeleteArrayListIterator(artsArrayListIterator * iter)
    {
        artsFree(iter);
    }
