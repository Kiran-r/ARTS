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
#ifndef ARTS_TERMINATION_DETECTION_H
#define  ARTS_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"

artsEpoch_t * createEpoch(artsGuid_t * guid, artsGuid_t edtGuid, unsigned int slot);
void incrementQueueEpoch(artsGuid_t epochGuid);
void incrementActiveEpoch(artsGuid_t epochGuid);
void incrementFinishedEpoch(artsGuid_t epochGuid);
void sendEpoch(artsGuid_t epochGuid, unsigned int source, unsigned int dest);
void broadcastEpochRequest(artsGuid_t epochGuid);
bool checkEpoch(artsEpoch_t * epoch, unsigned int totalActive, unsigned int totalFinish);
void reduceEpoch(artsGuid_t epochGuid, unsigned int active, unsigned int finish);
void deleteEpoch(artsGuid_t epochGuid, artsEpoch_t * epoch);

typedef struct artsEpochPool {
    struct artsEpochPool * next;
    unsigned int size;
    unsigned int index;
    volatile unsigned int outstanding;
    artsEpoch_t pool[];
} artsEpochPool_t;

artsEpochPool_t * createEpochPool(artsGuid_t * epochPoolGuid, unsigned int poolSize, artsGuid_t * startGuid);
artsEpoch_t * getPoolEpoch(artsGuid_t edtGuid, unsigned int slot);

void globalShutdownGuidIncActive();
void globalShutdownGuidIncQueue();
void globalShutdownGuidIncFinished();
bool createShutdownEpoch();

#ifdef __cplusplus
}
#endif
#endif
