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
#ifndef ARTSREMOTEPROTOCOL_H
#define ARTSREMOTEPROTOCOL_H
#ifdef __cplusplus
extern "C" {
#endif
//#define SEQUENCENUMBERS 1

//TODO: Switch to an enum
enum artsServerMessageType
{
    ARTS_REMOTE_SHUTDOWN_MSG=0,
    ARTS_REMOTE_EDT_SIGNAL_MSG,
    ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG,
    ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG,
    ARTS_REMOTE_ADD_DEPENDENCE_MSG,
    ARTS_REMOTE_DB_REQUEST_MSG,
    ARTS_REMOTE_DB_SEND_MSG,
    ARTS_REMOTE_INVALIDATE_DB_MSG,
    ARTS_REMOTE_DB_UPDATE_GUID_MSG,
    ARTS_REMOTE_DB_UPDATE_MSG,
    ARTS_REMOTE_DB_DESTROY_MSG,
    ARTS_REMOTE_DB_DESTROY_FORWARD_MSG,
    ARTS_REMOTE_DB_CLEAN_FORWARD_MSG,
    ARTS_REMOTE_DB_MOVE_REQ_MSG,      
    ARTS_REMOTE_EDT_MOVE_MSG,
    ARTS_REMOTE_EVENT_MOVE_MSG,
    ARTS_REMOTE_DB_MOVE_MSG,
    ARTS_REMOTE_PINGPONG_TEST_MSG,
    ARTS_REMOTE_METRIC_UPDATE_MSG,
    ARTS_REMOTE_DB_FULL_REQUEST_MSG,
    ARTS_REMOTE_DB_FULL_SEND_MSG,
    ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG,
    ARTS_REMOTE_GET_FROM_DB_MSG,
    ARTS_REMOTE_PUT_IN_DB_MSG,
    ARTS_REMOTE_SEND_MSG,
    ARTS_EPOCH_INIT_MSG,
    ARTS_EPOCH_INIT_POOL_MSG,
    ARTS_EPOCH_REQ_MSG, 
    ARTS_EPOCH_SEND_MSG,
    ARTS_EPOCH_DELETE_MSG,
    ARTS_ATOMIC_ADD_ARRAYDB_MSG,
    ARTS_ATOMIC_CAS_ARRAYDB_MSG,
    ARTS_REMOTE_BUFFER_SEND_MSG,
    ARTS_REMOTE_CONTEXT_SIG_MSG
};

//Header
struct __attribute__ ((__packed__)) artsRemotePacket
{
    unsigned int messageType;
    unsigned int size;
    unsigned int rank;
#ifdef SEQUENCENUMBERS
    unsigned int seqRank;
    uint64_t seqNum;
#endif
#ifdef COUNT
    uint64_t timeStamp;
    uint64_t procTimeStamp;
#endif
};

struct artsRemoteGuidOnlyPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteInvalidateDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteUpdateDbGuidPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteUpdateDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteMemoryMovePacket
{
    struct artsRemotePacket header;
    artsGuid_t guid;
};

struct __attribute__ ((__packed__)) artsRemoteAddDependencePacket
{
    struct artsRemotePacket header;
    artsGuid_t source;
    artsGuid_t destination;
    uint32_t slot;
    artsType_t mode;
    unsigned int destRoute;
};

struct __attribute__ ((__packed__)) artsRemoteEdtSignalPacket
{
    struct artsRemotePacket header;
    artsGuid_t edt;
    artsGuid_t db;
    uint32_t slot;
    artsType_t mode;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) artsRemoteEventSatisfySlotPacket
{
    struct artsRemotePacket header;
    artsGuid_t event;
    artsGuid_t db;
    uint32_t slot;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) artsRemoteDbRequestPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteDbSendPacket
{
    struct artsRemotePacket header;
};

struct __attribute__ ((__packed__)) artsRemoteDbFullRequestPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    void * edt;
    unsigned int slot;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteDbFullSendPacket
{
    struct artsRemotePacket header;
    struct artsEdt * edt;
    unsigned int slot;
    artsType_t mode;
};

struct __attribute__ ((__packed__)) artsRemoteMetricUpdate
{
    struct artsRemotePacket header;
    int type;
    uint64_t timeStamp;
    uint64_t toAdd;
    bool sub;
};

struct __attribute__ ((__packed__)) artsRemoteGetPutPacket
{
    struct artsRemotePacket header;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct __attribute__ ((__packed__)) artsRemoteSignalEdtWithPtrPacket
{
    struct artsRemotePacket header;
    artsGuid_t edtGuid;
    artsGuid_t dbGuid;
    unsigned int size;
    unsigned int slot;
};

struct __attribute__ ((__packed__)) artsRemoteSend
{
    struct artsRemotePacket header;
    sendHandler_t funPtr;
};

struct __attribute__ ((__packed__)) artsRemoteEpochInitPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
    artsGuid_t edtGuid;
    unsigned int slot;
};

struct __attribute__ ((__packed__)) artsRemoteEpochInitPoolPacket
{
    struct artsRemotePacket header;
    unsigned int poolSize;
    artsGuid_t startGuid;
    artsGuid_t poolGuid;
};

struct __attribute__ ((__packed__)) artsRemoteEpochReqPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
};

struct __attribute__ ((__packed__)) artsRemoteEpochSendPacket
{
    struct artsRemotePacket header;
    artsGuid_t epochGuid;
    unsigned int active;
    unsigned int finish;
};

struct __attribute__ ((__packed__)) artsRemoteAtomicAddInArrayDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int toAdd;
};

struct __attribute__ ((__packed__)) artsRemoteAtomicCompareAndSwapInArrayDbPacket
{
    struct artsRemotePacket header;
    artsGuid_t dbGuid;
    artsGuid_t edtGuid;
    artsGuid_t epochGuid;
    unsigned int slot;
    unsigned int index;
    unsigned int oldValue;
    unsigned int newValue;
};

struct __attribute__ ((__packed__)) artsRemoteSignalContextPacket
{
    struct artsRemotePacket header;
    uint64_t ticket;
};

void outInit( unsigned int size );
bool artsRemoteAsyncSend();
void artsRemoteSendRequestAsync( int rank, char * message, unsigned int length );
void artsRemoteSendRequestAsyncEnd( int rank, char * message, unsigned int length );
void artsRemoteSendRequestPayloadAsync( int rank, char * message, unsigned int length, char * payload, unsigned int size );
void artsRemoteSendRequestPayloadAsyncFree( int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, artsGuid_t guid, void(*freeMethod)(void*));
void artsRemoteSendRequestPayloadAsyncCopy( int rank, char * message, unsigned int length, char * payload, unsigned int size );
void artsRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop);
#ifdef __cplusplus
}
#endif

#endif
