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
#ifndef ARTSGLOBALS_H
#define ARTSGLOBALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "artsArrayList.h"
#include "artsCounter.h"
#include "artsQueue.h"

struct atomicCreateBarrierInfo
{
    volatile unsigned int wait;
    volatile unsigned int result;
};

struct artsRuntimeShared
{
    volatile unsigned int sendLock;
    char pad1[56];
    volatile unsigned int recvLock;
    char pad2[56];
    volatile unsigned int stealRequestLock;
    char pad3[56];
    bool (*scheduler)();
    struct artsDeque ** deque;
    struct artsDeque ** receiverDeque;
    struct artsRouteTable ** routeTable;
    struct artsRouteTable * remoteRouteTable;
    volatile bool ** localSpin;
    volatile bool ** tMTLocalSpin;
    unsigned int ** memoryMoves;
    struct atomicCreateBarrierInfo ** atomicWaits;
    unsigned int workerThreadCount;
    unsigned int senderThreadCount;
    unsigned int receiverThreadCount;
    unsigned int remoteStealingThreadCount;
    unsigned int totalThreadCount;
    volatile unsigned int readyToPush;
    volatile unsigned int readyToParallelStart;
    volatile unsigned int readyToInspect;
    volatile unsigned int readyToExecute;
    volatile unsigned int readyToClean;
    volatile unsigned int readyToShutdown;
    char * buf;
    int packetSize;
    bool shutdownStarted;
    volatile unsigned int shutdownCount;
    uint64_t shutdownTimeout;
    uint64_t shutdownForceTimeout;
    unsigned int printNodeStats;
    artsGuid_t shutdownEpoch;
    unsigned int shadLoopStride;
    bool tMT;
    uint64_t ** keys;
    uint64_t * globalGuidThreadId;
}__attribute__ ((aligned(64)));

struct artsRuntimePrivate
{
    struct artsDeque * myDeque;
    struct artsDeque * myNodeDeque;
    unsigned int coreId;
    unsigned int threadId;
    unsigned int groupId;
    unsigned int clusterId;
    unsigned int backOff;
    volatile unsigned int oustandingMemoryMoves;
    struct atomicCreateBarrierInfo atomicWait;
    volatile bool alive;
    volatile bool worker;
    volatile bool networkSend;
    volatile bool networkReceive;
    volatile bool statusSend;
    artsGuid_t currentEdtGuid;
    int mallocType;
    int mallocTrace;
    int edtFree;
    int localCounting;
    unsigned int shadLock;
    artsArrayList * counterList;
    unsigned short drand_buf[3];
};

extern struct artsRuntimeShared artsNodeInfo;
extern __thread struct artsRuntimePrivate artsThreadInfo;

extern unsigned int artsGlobalRankId;
extern unsigned int artsGlobalRankCount;
extern unsigned int artsGlobalMasterRankId;
extern bool artsGlobalIWillPrint;
extern uint64_t artsGuidMin;
extern uint64_t artsGuidMax;

#define MASTER_PRINTF(...) if (artsGlobalRankId==artsGlobalMasterRankId) PRINTF(__VA_ARGS__)
#define ONCE_PRINTF(...) if(artsGlobalIWillPrint == true) PRINTF(__VA_ARGS__)

#define artsTypeName const char * const _artsTypeName[] = { \
"ARTS_NULL", \
"ARTS_EDT", \
"ARTS_EVENT", \
"ARTS_EPOCH", \
"ARTS_CALLBACK", \
"ARTS_BUFFER", \
"ARTS_DB_READ", \
"ARTS_DB_WRITE", \
"ARTS_DB_PIN", \
"ARTS_DB_ONCE", \
"ARTS_DB_ONCE_LOCAL", \
"ARTS_LAST_TYPE", \
"ARTS_SINGLE_VALUE", \
"ARTS_PTR" }

#define getTypeName(x) _artsTypeName[x]

extern const char * const _artsTypeName[];

extern volatile uint64_t outstandingEdts;
void checkOutEdts(uint64_t threashold);

#ifdef CHECK_NO_EDT
#define incOustandingEdts(numEdts) artsAtomicFetchAddU64(&outstandingEdts, numEdts)
#define decOustandingEdts(numEdts) artsAtomicFetchSubU64(&outstandingEdts, numEdts)
#define checkOutstandingEdts(threashold) checkOutEdts(threashold)
#else
#define incOustandingEdts(numEdts)
#define decOustandingEdts(numEdts)
#define checkOutstandingEdts(threashold)
#endif

#ifdef __cplusplus
}
#endif

#endif
