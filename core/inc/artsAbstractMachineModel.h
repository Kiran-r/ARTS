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
#ifndef ARTSABSTRACTMACHINEMODEL_H
#define	ARTSABSTRACTMACHINEMODEL_H
#ifdef __cplusplus
extern "C" {
#endif
#include <assert.h>
#include "arts.h"
#include "artsConfig.h"
    
struct artsCoreInfo;
struct unitThread
{
    unsigned int id;
    unsigned int groupId;
    unsigned int groupPos;
    bool worker;
    bool networkSend;
    bool networkReceive;
    bool statusSend;
    bool pin;
    struct unitThread * next;
};

struct threadMask
{
    unsigned int clusterId;
    unsigned int coreId;
    unsigned int unitId;
    bool on;
    unsigned int id;
    unsigned int groupId;
    unsigned int groupPos;
    bool worker;
    bool networkSend;
    bool networkReceive;
    bool statusSend;
    bool pin;
    struct artsCoreInfo * coreInfo;
};

struct unitMask
{
    unsigned int clusterId;
    unsigned int coreId;
    unsigned int unitId;
    bool on;
    unsigned int threads;
    struct unitThread * listHead;
    struct unitThread * listTail;
    struct artsCoreInfo * coreInfo;
};

struct coreMask
{
    unsigned int numUnits;
    struct unitMask * unit;
};

struct clusterMask 
{
    unsigned int numCores;
    struct coreMask * core;
};

struct nodeMask
{
    unsigned int numClusters;
    struct clusterMask * cluster;
};
    
struct threadMask * getThreadMask(struct artsConfig * config);
void printMask(struct threadMask * units, unsigned int numberOfUnits);
void artsAbstractMachineModelPinThread(struct artsCoreInfo * coreInfo);

#ifdef __cplusplus
}
#endif

#endif	/* artsABSTRACTMACHINEMODEL_H */

