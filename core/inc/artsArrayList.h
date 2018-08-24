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
#ifndef ARTSARRAYLIST_H
#define	ARTSARRAYLIST_H
#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif	/* LIST_H */

