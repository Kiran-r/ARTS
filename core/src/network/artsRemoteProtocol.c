#include "arts.h"
#include "artsLinkList.h"
#include "artsGlobals.h"
#include "artsMalloc.h"
#include "artsAtomics.h"
#include "artsIntrospection.h"
#include "artsRemoteProtocol.h"
#include "artsCounter.h"
#include <string.h>
#include "artsDebug.h"
#include "artsRemote.h"
#include "artsServer.h"
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
    artsGuid_t nullMe;
    bool end;
};

unsigned int nodeListSize;

struct artsLinkList * outHead;
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

void artsRemotSetThreadOutboundQueues(unsigned int start, unsigned int stop)
{
    threadStart = start;
    threadStop = stop;
    
    unsigned int size = stop - start;
    outResend = artsCalloc(sizeof(struct outList *)*size);
    outResendFree = artsCalloc(sizeof(void *)*size);
}

void outInit( unsigned int size )
{
    nodeListSize = size;
    outHead = artsLinkListGroupNew(size);
#ifdef SEQUENCENUMBERS
    seqNumber = artsCalloc(sizeof(u64)*artsGlobalRankCount);
    seqNumLock = artsCalloc(sizeof(unsigned int)*size);
#endif
}

static inline void outInsertNode( struct outList * node, unsigned int length  )
{
#ifdef COUNT
    //This is for network queue sitting time...
//    node->timeStamp = artsExtGetTimeStamp();
#endif
    
    //int listId = node->rank*ports+artsThreadInfo.threadId%ports;
    long unsigned int listId;
    //mrand48_r (&artsThreadInfo.drand_buf, &listId);
    listId = node->rank*ports + artsThreadInfo.groupId % ports;
    struct artsLinkList * list = artsLinkListGet(outHead, listId);
    struct artsRemotePacket *packet = (struct artsRemotePacket *) (node + 1);
#ifdef SEQUENCENUMBERS
    artsLock(&seqNumLock[listId]);
    packet->seqNum = artsAtomicFetchAddU64(&seqNumber[node->rank], 1U);
    packet->seqRank = artsGlobalRankId;
#endif
    artsLinkListPushBack(list, node, length);
#ifdef SEQUENCENUMBERS
    artsUnlock(&seqNumLock[listId]);
#endif
//    artsUpdatePerformanceMetric(artsNetworkQueuePush, artsThread, packet->size, false);
    artsUpdatePerformanceMetric(artsNetworkQueuePush, artsThread, 1, false);
}

static inline struct outList * outPopNode( unsigned int threadId, void ** freeMe  )
{
    struct outList * out;
    struct artsLinkList * list;
    list = artsLinkListGet(outHead, threadId);
    out = artsLinkListPopFront( list, freeMe );
    if(out)
    {
        struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
//        artsUpdatePerformanceMetric(artsNetworkQueuePop, artsThread, packet->size, false);
    }
    artsUpdatePerformanceMetric(artsNetworkQueuePop, artsThread, 1, false);
    return out;
}

bool sendEnd = false;
extern int lastRank;

bool artsRemoteAsyncSend() 
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
            artsRemoteCheckProgress(i);
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
                if (out->rank == artsGlobalRankId) 
                {
                    struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
                    PRINTF("%d %d Self send error\n", artsGlobalRankId, packet->messageType);
                }
                if (out->payload == NULL) 
                {
                    lengthRemaining = artsRemoteSendRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length);
                } 
                else 
                {
                    lengthRemaining = artsRemoteSendPayloadRequest(out->rank, i, ((char*) (out + 1))+out->offset, out->length, ((char *)out->payload)+out->offsetPayload, out->payloadSize);
                    
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
                    struct artsRemotePacket *packet = (struct artsRemotePacket *) (out + 1);
                    if(packet->messageType!=ARTS_REMOTE_METRIC_UPDATE_MSG)
                        artsUpdatePerformanceMetric(artsNetworkSendBW, artsThread, packet->size, false);
                    
                    outResend[i-threadStart] = NULL;
                    artsFree(freeMe);
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
    if(rank == artsGlobalRankId || rank >= artsGlobalRankCount  )
    {
        PRINTF("Send error rank stack trace: %u of %u\n", rank, artsGlobalRankCount); 
        artsDebugPrintStack();
        artsDebugGenerateSegFault();
    }
}

static inline void sizeSendCheck( unsigned int size )
{
    if( size == 0 || size > 1073741824)
    {
        PRINTF("Send error size stack trace: %d\n", size); 
        artsDebugPrintStack();
        artsDebugGenerateSegFault();
    }
}

static inline bool lockNetwork( volatile unsigned int * lock)
{

    if(*lock == 0U)
    {
        if(artsAtomicCswap( lock, 0U, artsThreadInfo.threadId+1U ) == 0U)
            return true;
    }
    
    return false;
}

static inline void unlockNetwork( volatile unsigned int * lock)
{
    //artsAtomicSwap( lock, 0U );
    *lock=0U;
}

void artsRemoteSendRequestAsyncEnd( int rank, char * message, unsigned int length )
{
    selfSendCheck(rank); 
    struct outList * next = artsLinkListNewItem( length + sizeof(struct outList) );
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

void artsRemoteSendRequestAsync( int rank, char * message, unsigned int length )
{
    selfSendCheck(rank); 
    struct outList * next = artsLinkListNewItem( length + sizeof(struct outList) );
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

void artsRemoteSendRequestPayloadAsync( int rank, char * message, unsigned int length, char * payload, unsigned int size )
{
    selfSendCheck(rank); 
    sizeSendCheck( length );
    sizeSendCheck( size);
    struct outList * next = artsLinkListNewItem( length + sizeof(struct outList) );
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

void artsRemoteSendRequestPayloadAsyncFree(int rank, char * message, unsigned int length, char * payload, unsigned int offset, unsigned int size, artsGuid_t guid, void(*freeMethod)(void*))
{
    selfSendCheck(rank); 
    sizeSendCheck( length );
    sizeSendCheck( size);
    struct outList * next = artsLinkListNewItem( length + sizeof(struct outList) );
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

void artsRemoteSendRequestPayloadAsyncCopy( int rank, char * message, unsigned int length, char * payload, unsigned int size )
{
    selfSendCheck(rank); 
    sizeSendCheck( length + size );
    struct outList * next = artsLinkListNewItem( length + size + sizeof(struct outList) );
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
