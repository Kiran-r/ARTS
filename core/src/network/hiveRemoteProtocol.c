#include "hive.h"
#include "hiveLinkList.h"
#include "hiveGlobals.h"
#include "hiveMalloc.h"
#include "hiveAtomics.h"
#include "hiveIntrospection.h"
#include "hiveRemoteProtocol.h"
#include "hiveCounter.h"
#include <string.h>
#include "hiveDebug.h"
#include "hiveRemote.h"
#include "hiveServer.h"
#define DPRINTF( ... )

struct outList
{
#ifdef COUNT
    uint64_t timeStamp;
#endif
    unsigned int offset;
    unsigned int length;
    unsigned int rank;
    void * payload;
    unsigned int payloadSize;
    unsigned int offsetPayload;
    void (*freeMethod)(void *);
    hiveGuid_t nullMe;
    bool end;
};

unsigned int nodeListSize;

struct hiveLinkList * outHead;
extern unsigned int ports;

__thread unsigned int threadStart;
__thread unsigned int threadStop;

__thread struct outList ** outResend;
__thread void ** outResendFree;

#ifdef SEQUENCENUMBERS
unsigned int * seqNumLock = NULL;
u64 * seqNumber = NULL;
#endif

void partialSendStore(struct outList * out , unsigned int lengthRemaining)
{
    //PRINTF("Payload\n");
    if (out->payload == NULL)
    {
        out->offset = out->offset+(out->length-lengthRemaining);
        out->length = lengthRemaining;
    }
    else
    {
        unsigned int sent = out->length+out->payloadSize;
        sent -= lengthRemaining; 
        //PRINTF("Here %d\n", sent);
        if(sent >= out->length )
        {
            out->length=0;
            out->offsetPayload = out->offsetPayload + (out->payloadSize-lengthRemaining);
            out->payloadSize = lengthRemaining;
            //PRINTF("Here 1 %d %d %d %p\n", out->payloadSize, out->offsetPayload, lengthRemaining, out );
            
        }
        else
        {
            //PRINTF("Here 2a %d %d %d %d\n", out->payloadSize, out->length, sent, lengthRemaining);
            out->offset = out->offset + (out->length-(lengthRemaining-out->payloadSize));
            out->length = lengthRemaining - out->payloadSize;
            //PRINTF("Here 2b %d %d\n", out->offset, out->length);
        }
    }
    
}

void hiveRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop)
{
    threadStart = start;
    threadStop = stop;
    
    unsigned int size = stop - start;
    outResend = hiveCalloc(sizeof(struct outList *)*size);
    outResendFree = hiveCalloc(sizeof(void *)*size);
}

void outInit( unsigned int size )
{
    nodeListSize = size;
    outHead = hiveLinkListGroupNew(size);
#ifdef SEQUENCENUMBERS
    seqNumber = hiveCalloc(sizeof(u64)*hiveGlobalRankCount);
    seqNumLock = hiveCalloc(sizeof(unsigned int)*size);
#endif
}

static inline void outInsertNode( struct outList * node, unsigned int length  )
{
#ifdef COUNT
    //This is for network queue sitting time...
    node->timeStamp = hiveExtGetTimeStamp();
#endif
    
    //int listId = node->rank*ports+hiveThreadInfo.threadId%ports;
    long unsigned int listId;
    //mrand48_r (&hiveThreadInfo.drand_buf, &listId);
    listId = node->rank*ports + hiveThreadInfo.groupId % ports;
    struct hiveLinkList * list = hiveLinkListGet(outHead, listId);
    struct hiveRemotePacket *packet = (struct hiveRemotePacket *) (node + 1);
#ifdef SEQUENCENUMBERS
    hiveLock(&seqNumLock[listId]);
    packet->seqNum = hiveAtomicFetchAddU64(&seqNumber[node->rank], 1U);
    packet->seqRank = hiveGlobalRankId;
#endif
    hiveLinkListPushBack(list, node, length);
#ifdef SEQUENCENUMBERS
    hiveUnlock(&seqNumLock[listId]);
#endif
//    hiveUpdatePerformanceMetric(hiveNetworkQueuePush, hiveThread, packet->size, false);
    hiveUpdatePerformanceMetric(hiveNetworkQueuePush, hiveThread, 1, false);
}

static inline struct outList * outPopNode( unsigned int threadId, void ** freeMe  )
{
    struct outList * out;
    struct hiveLinkList * list;
    list = hiveLinkListGet(outHead, threadId);
    out = hiveLinkListPopFront( list, freeMe );
    if(out)
    {
        struct hiveRemotePacket *packet = (struct hiveRemotePacket *) (out + 1);
//        hiveUpdatePerformanceMetric(hiveNetworkQueuePop, hiveThread, packet->size, false);
    }
    hiveUpdatePerformanceMetric(hiveNetworkQueuePop, hiveThread, 1, false);
    return out;
}

bool sendEnd = false;
extern int lastRank;

bool hiveRemoteAsyncSend() 
{
    int i;
    bool success = false;
    bool sent = true;
    void * freeMe;
    unsigned int lengthRemaining;
    struct outList * out; 
    while (sent) 
    {
        sent = false;
        for (i = threadStart; i < threadStop; i++) 
        {
            out = NULL;
            lastRank =i;
            hiveRemoteCheckProgress(i);
            if(outResend[i-threadStart])
            {
                out = outResend[i-threadStart];
                freeMe = outResendFree[i-threadStart];
            }
            else
            {
                if(!sendEnd)
                    out = outPopNode(i, &freeMe);
                if(out && out->end)
                {
                    sendEnd = true;
                }
            }
            if (out != NULL) 
            {
                DPRINTF("KSending %p %p %d\n", out, freeMe, out->rank);
                if (out->rank == hiveGlobalRankId) 
                {
                    struct hiveRemotePacket *packet = (struct hiveRemotePacket *) (out + 1);
                    PRINTF("%d %d Self send error\n", hiveGlobalRankId, packet->messageType);
                }
                if (out->payload == NULL) 
                {
                    lengthRemaining = hiveRemoteSendRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length);
                } 
                else 
                {
                    lengthRemaining = hiveRemoteSendPayloadRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length, ((char *)out->payload)+out->offsetPayload, out->payloadSize);
                    
                    //if (out->freeMethod)
                    if (out->freeMethod && !lengthRemaining)
                    {
                        DPRINTF("Null Me %ld\n", out->nullMe);
                        out->freeMethod(out->payload);
                    }
                }
                DPRINTF("KSending Done %p %p %d\n", out, freeMe, out->rank);
                
                if(lengthRemaining == -1)
                {
                    return false;
                }

                if(lengthRemaining)
                {
//                    PRINTF("Here %d\n", lengthRemaining);
                    //outResend[i-threadStart] = NULL;
                    partialSendStore(out,lengthRemaining);
                    outResend[i-threadStart] = out;
                    outResendFree[i-threadStart] = freeMe;
                }
                else
                {
                    struct hiveRemotePacket *packet = (struct hiveRemotePacket *) (out + 1);
                    if(packet->messageType!=HIVE_REMOTE_METRIC_UPDATE_MSG)
                        hiveUpdatePerformanceMetric(hiveNetworkSendBW, hiveThread, packet->size, false);
                    
                    outResend[i-threadStart] = NULL;
                    hiveFree(freeMe);
                }
                sent = true;
                success = true;
            }
        }
    }
    return success;
}

static inline void selfSendCheck( unsigned int rank )
{
    if(rank == hiveGlobalRankId || rank >= hiveGlobalRankCount  )
    {
        PRINTF("Send error rank stack trace: %u of %u\n", rank, hiveGlobalRankCount); 
        hiveDebugPrintStack();
        hiveDebugGenerateSegFault();
    }
}

static inline void sizeSendCheck( unsigned int size )
{
    if( size == 0 || size > 1073741824)
    {
        PRINTF("Send error size stack trace: %d\n", size); 
        hiveDebugPrintStack();
        hiveDebugGenerateSegFault();
    }
}

static inline bool lockNetwork( volatile unsigned int * lock)
{

    if(*lock == 0U)
    {
        if(hiveAtomicCswap( lock, 0U, hiveThreadInfo.threadId+1U ) == 0U)
            return true;
    }
    
    return false;
}

static inline void unlockNetwork( volatile unsigned int * lock)
{
    //hiveAtomicSwap( lock, 0U );
    *lock=0U;
}

void hiveRemoteSendRequestAsyncEnd( int rank, char * message, unsigned int length )
{
    selfSendCheck(rank); 
    struct outList * next = hiveLinkListNewItem( length + sizeof(struct outList) );
    next->offset=0;
    next->offsetPayload=0;
    next->length = length;
    next->rank = rank;
    next->payload=NULL;
    next->end=true;
    memcpy( next+1, message, length );
    DPRINTF("KAdding %p \n", next);
    outInsertNode( next, length+sizeof(struct outList) );
}

void hiveRemoteSendRequestAsync( int rank, char * message, unsigned int length )
{
    selfSendCheck(rank); 
    struct outList * next = hiveLinkListNewItem( length + sizeof(struct outList) );
    next->offset=0;
    next->offsetPayload=0;
    next->length = length;
    next->rank = rank;
    next->payload=NULL;
    next->end=false;
    memcpy( next+1, message, length );
    DPRINTF("KAdding %p \n", next);
    outInsertNode( next, length+sizeof(struct outList) );
}

void hiveRemoteSendRequestPayloadAsync( int rank, char * message, unsigned int length, char * payload, unsigned int size )
{
    selfSendCheck(rank); 
    sizeSendCheck( length );
    sizeSendCheck( size);
    struct outList * next = hiveLinkListNewItem( length + sizeof(struct outList) );
    next->offset=0;
    next->offsetPayload=0;
    next->length = length;
    next->rank = rank;
    next->payload=payload;
    next->freeMethod = NULL;
    next->payloadSize=size;
    next->end=false;
    memcpy( next+1, message, length );
    outInsertNode( next, length+sizeof(struct outList) );
}

void hiveRemoteSendRequestPayloadAsyncFree(int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, hiveGuid_t guid, void(*freeMethod)(void*))
{
    selfSendCheck(rank); 
    sizeSendCheck( length );
    sizeSendCheck( size);
    struct outList * next = hiveLinkListNewItem( length + sizeof(struct outList) );
    next->offset=0;
    next->offsetPayload=offset;
    next->length = length;
    next->rank = rank;
    next->payload=payload;
    next->payloadSize=size;
    next->freeMethod = freeMethod;
    next->nullMe = guid;
    next->end=false;
    memcpy( next+1, message, length );
    outInsertNode( next, length+sizeof(struct outList) );
}

void hiveRemoteSendRequestPayloadAsyncCopy( int rank, char * message, unsigned int length, char * payload, unsigned int size )
{
    selfSendCheck(rank); 
    sizeSendCheck( length + size );
    struct outList * next = hiveLinkListNewItem( length + size + sizeof(struct outList) );
    next->offset=0;
    next->offsetPayload=0;
    next->length = length+size;
    next->rank = rank;
    next->payload=NULL;
    next->end=false;
    memcpy( next+1, message, length );
    memcpy( ((char *)(next+1))+length, payload, size );
    outInsertNode( next, size+length+sizeof(struct outList) );
}
