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
#ifndef ARTSDEQUE_H
#define ARTSDEQUE_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"  
#define STEALSIZE 1024

struct artsDeque;
struct artsDeque *artsDequeListNew(unsigned int listSize, unsigned int dequeSize);
struct artsDeque *artsDequeListGetDeque(struct artsDeque *dequeList, unsigned int position);
void artsDequeListDelete(void *dequeList);
struct artsDeque * artsDequeNew(unsigned int size);
void artsDequeDelete(struct artsDeque *deque);
bool artsDequePushFront(struct artsDeque *deque, void *item, unsigned int priority);
void *artsDequePopFront(struct artsDeque *deque);
void *artsDequePopBack(struct artsDeque *deque);

bool artsDequeEmpty(struct artsDeque *deque);
void artsDequeClear(struct artsDeque *deque);
unsigned int artsDequeSize(struct artsDeque *deque);

#ifdef __cplusplus
}
#endif
#endif
