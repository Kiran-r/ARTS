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
#ifndef ARTSEDTFUNCTIONS_H
#define ARTSEDTFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

bool artsEdtCreateInternal(artsGuid_t * guid, unsigned int route, unsigned int cluster, unsigned int edtSpace, artsGuid_t eventGuid, artsEdt_t funcPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, bool useEpoch, artsGuid_t epochGuid, bool hasDepv);
void artsEdtDelete(struct artsEdt * edt);
void internalSignalEdt(artsGuid_t edtPacket, uint32_t slot, artsGuid_t dataGuid, artsType_t mode, void * ptr, unsigned int size);

typedef struct 
{
    artsGuid_t currentEdtGuid;
    struct artsEdt * currentEdt;
    void * epochList;
} threadLocal_t;

void artsSetThreadLocalEdtInfo(struct artsEdt * edt);
void artsUnsetThreadLocalEdtInfo();
void artsSaveThreadLocal(threadLocal_t * tl);
void artsRestoreThreadLocal(threadLocal_t * tl);


bool artsSetCurrentEpochGuid(artsGuid_t epochGuid);
artsGuid_t * artsCheckEpochIsRoot(artsGuid_t toCheck);
void artsIncrementFinishedEpochList();

typedef struct {
    void * buffer;
    uint32_t * sizeToWrite;
    unsigned int size;
    artsGuid_t epochGuid;
    volatile unsigned int uses;
} artsBuffer_t;

#ifdef __cplusplus
}
#endif

#endif
