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
