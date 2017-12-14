#ifndef HIVEREMOTEPROTOCOL_H
#define HIVEREMOTEPROTOCOL_H

//#define SEQUENCENUMBERS 1

//TODO: Switch to an enum
enum hiveServerMessageType
{
    HIVE_REMOTE_SHUTDOWN_MSG=0,
    HIVE_REMOTE_EDT_STEAL_MSG,
    HIVE_REMOTE_EDT_RECV_MSG,
    HIVE_REMOTE_EDT_FAIL_MSG,
    HIVE_REMOTE_EDT_SIGNAL_MSG,
    HIVE_REMOTE_EVENT_SATISFY_MSG,
    HIVE_REMOTE_EVENT_SATISFY_SLOT_MSG,
    HIVE_REMOTE_DB_REQUEST_MSG,
    HIVE_REMOTE_DB_SEND_MSG,
    HIVE_REMOTE_EDT_MOVE_MSG,
    HIVE_REMOTE_GUID_ROUTE_MSG,
    HIVE_REMOTE_EVENT_MOVE_MSG,
    HIVE_REMOTE_ADD_DEPENDENCE_MSG,
    HIVE_REMOTE_INVALIDATE_DB_MSG,
    HIVE_REMOTE_DB_MOVE_MSG,
    HIVE_REMOTE_DB_UPDATE_GUID_MSG,
    HIVE_REMOTE_DB_UPDATE_MSG,
    HIVE_REMOTE_DB_DESTROY_MSG,
    HIVE_REMOTE_DB_DESTROY_FORWARD_MSG,
    HIVE_REMOTE_DB_CLEAN_FORWARD_MSG,
    HIVE_REMOTE_PINGPONG_TEST_MSG,
    HIVE_DB_LOCK_MSG,
    HIVE_DB_UNLOCK_MSG,
    HIVE_DB_LOCK_ALL_DBS_MSG,
    HIVE_REMOTE_METRIC_UPDATE,
    HIVE_ACTIVE_MESSAGE,
    HIVE_REMOTE_DB_FULL_REQUEST_MSG,
    HIVE_REMOTE_DB_FULL_SEND_MSG,
    HIVE_REMOTE_DB_FULL_SEND_ALREADY_LOCAL,
    HIVE_REMOTE_GET_FROM_DB,
    HIVE_REMOTE_PUT_IN_DB,
    HIVE_REMOTE_SIGNAL_EDT_WITH_PTR,
    HIVE_REMOTE_SEND
};

//Header
struct __attribute__ ((__packed__)) hiveRemotePacket
{
    unsigned int messageType;
    unsigned int size;
    unsigned int rank;
#ifdef SEQUENCENUMBERS
    unsigned int seqRank;
    u64 seqNum;
#endif
#ifdef COUNT
    u64 timeStamp;
    u64 procTimeStamp;
#endif
};

struct __attribute__ ((__packed__)) hiveDbLockAllDbsPacket
{
    struct hiveRemotePacket header;
    void * edt;
};

struct __attribute__ ((__packed__)) hiveDbUnlockPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t dbGuid;
    bool write;
};

struct __attribute__ ((__packed__)) hiveDbLockPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t dbGuid;
    void * edt;
    bool shared;
};

struct hiveRemoteGuidOnlyPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
};

struct __attribute__ ((__packed__)) hiveRemoteInvalidateDbPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
};

struct __attribute__ ((__packed__)) hiveRemoteUpdateDbGuidPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
};

struct __attribute__ ((__packed__)) hiveRemoteUpdateDbPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
};

struct __attribute__ ((__packed__)) hiveRemotePingBackPacket
{
    struct hiveRemotePacket header;
    void * signalMe;
};

struct __attribute__ ((__packed__)) hiveRemoteRouteGuidPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
    unsigned int route;
};

struct __attribute__ ((__packed__)) hiveRemoteMemoryMovePacket
{
    struct hiveRemotePacket header;
    hiveGuid_t guid;
};

struct __attribute__ ((__packed__)) hiveRemoteAddDependencePacket
{
    struct hiveRemotePacket header;
    hiveGuid_t source;
    hiveGuid_t destination;
    u32 slot;
    hiveDbAccessMode_t mode;

    unsigned int destRoute;
};

struct __attribute__ ((__packed__)) hiveRemoteEdtSignalPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t edt;
    hiveGuid_t db;
    u32 slot;
    hiveDbAccessMode_t mode;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) hiveRemoteEventSatisfyPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t event;
    hiveGuid_t db;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) hiveRemoteEventSatisfySlotPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t event;
    hiveGuid_t db;
    u32 slot;
    //-------------------------Routing info
    unsigned int dbRoute;
};

struct __attribute__ ((__packed__)) hiveRemoteDbRequestPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t dbGuid;
    hiveDbAccessMode_t mode;
};

struct __attribute__ ((__packed__)) hiveRemoteDbSendPacket
{
    struct hiveRemotePacket header;
};

struct __attribute__ ((__packed__)) hiveRemoteDbFullRequestPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t dbGuid;
    void * edt;
    unsigned int slot;
    hiveDbAccessMode_t mode;
};

struct __attribute__ ((__packed__)) hiveRemoteDbFullSendPacket
{
    struct hiveRemotePacket header;
    struct hiveEdt * edt;
    unsigned int slot;
    hiveDbAccessMode_t mode;
};

struct __attribute__ ((__packed__)) hiveRemoteMemoryMovePingPacket
{
    struct hiveRemotePacket header;
    unsigned int threadToPing;
    unsigned int result;
};

struct __attribute__ ((__packed__)) hiveRemoteShutdownPingPacket
{
    struct hiveRemotePacket header;
};

struct __attribute__ ((__packed__)) hiveRemoteMetricUpdate
{
    struct hiveRemotePacket header;
    int type;
    u64 timeStamp;
    u64 toAdd;
    bool sub;
};

struct __attribute__ ((__packed__)) hiveRemoteGetPutPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t edtGuid;
    hiveGuid_t dbGuid;
    unsigned int slot;
    unsigned int offset;
    unsigned int size;
};

struct __attribute__ ((__packed__)) hiveRemoteSignalEdtWithPtrPacket
{
    struct hiveRemotePacket header;
    hiveGuid_t edtGuid;
    hiveGuid_t dbGuid;
    unsigned int size;
    unsigned int slot;
};

struct __attribute__ ((__packed__)) hiveRemoteSend
{
    struct hiveRemotePacket header;
    sendHandler_t funPtr;
};

void outInit( unsigned int size );
bool hiveRemoteAsyncSend();
void hiveRemoteSendRequestAsync( int rank, char * message, unsigned int length );
void hiveRemoteSendRequestAsyncEnd( int rank, char * message, unsigned int length );
void hiveRemoteSendRequestPayloadAsync( int rank, char * message, unsigned int length, char * payload, unsigned int size );
void hiveRemoteSendRequestPayloadAsyncFree( int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, hiveGuid_t guid, void(*freeMethod)(void*));
void hiveRemoteSendRequestPayloadAsyncCopy( int rank, char * message, unsigned int length, char * payload, unsigned int size );
void hiveRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop);
#endif
