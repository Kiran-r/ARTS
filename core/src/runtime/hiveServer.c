#include "hive.h"
#include "hiveRemoteProtocol.h"
#include "hiveGlobals.h"
#include "hiveRuntime.h"
#include "hiveServer.h"
#include "hiveRemoteFunctions.h"
#include "hiveEdtFunctions.h"
#include "hiveEventFunctions.h"
#include "hiveRouteTable.h"
#include "hiveRemote.h"
#include "hiveAtomics.h"
#include "hiveIntrospection.h"
#include "hiveTimer.h"
#include <unistd.h>
//#include "hiveRemote.h"
#define DPRINTF( ... )
//#define DPRINTF(...) PRINTF(__VA_ARGS__)

#define EDT_MUG_SIZE 32 

bool hiveGlobalIWillPrint=false;
FILE * lockPrintFile;
extern bool serverEnd;

#ifdef SEQUENCENUMBERS
u64 * recSeqNumbers;
#endif

void hiveRemoteTryToBecomePrinter()
{
    lockPrintFile = fopen(".hivePrintLock", "wx");
    if(lockPrintFile)
        hiveGlobalIWillPrint=true;
}

void hiveRemoteTryToClosePrinter()
{
    if(lockPrintFile)
        fclose(lockPrintFile);
    remove(".hivePrintLock");
}

bool hiveServerEnd()
{
    //PRINTF("Here\n");
    hiveNodeInfo.shutdownStarted=true;
    if(hiveRemoteShutdownSend())
    {
        //PRINTF("Not here\n");
        hiveRuntimeStop();
        return true;
    }

    hiveNodeInfo.shutdownTimeout = hiveGetTimeStamp()+500000000;
    return false;
}

void hiveRemoteShutdown()
{
    hiveLLServerShutdown();
}

void hiveServerSendStealRequest()
{
    static int node=0;
    
    if(node == hiveGlobalRankId)
        node++;
    
    if(node>=hiveGlobalRankCount)
    {
        if(hiveGlobalRankId != 0)
            node=0;
        else
            node=1;
    }
    
    DPRINTF("Steal to node %d\n", node);

    struct hiveRemotePacket packed;
    packed.rank = hiveGlobalRankId;
    packed.messageType = HIVE_REMOTE_EDT_STEAL_MSG;
    packed.size = sizeof(packed);
    hiveRemoteSendRequestAsync( node , (char *)&packed,  sizeof(packed) );
    node++;
}

static inline bool mugEdts(int requester, int mugSize)
{
    void * muggerCoat[EDT_MUG_SIZE];
    int stolen = hiveRuntimeStealAnyMultipleEdt( EDT_MUG_SIZE, muggerCoat );
    //void * muggerCoat;
    //muggerCoat[0] = hiveRuntimeStealFromWorker(0);
    //if(muggerCoat[0]!= NULL)
    //    stolen++;
    int i;
    if(stolen>0)
    {
        DPRINTF("stolen %d %d\n",stolen, requester);
        struct hiveRemotePacket * packed;
        int size = packageEdts(muggerCoat, stolen, (void **)&packed);
        //for(i=0; i< stolen; i++)
        //    hiveRouteTableInvalidateItem( NULL, (hiveGuid_t) ((struct hiveEdt *)muggerCoat[i])->currentEdt, requester, 1);
        hiveRemoteSendRequestAsync( requester, (char *)packed,  size);
        hiveFree(packed);
        for(i=0; i< stolen; i++)
        {
            struct hiveEdt * edt = (struct hiveEdt*)muggerCoat[i];
            hiveFree(muggerCoat[i]);
        }
        return true;
    }
    else
    {
        struct hiveRemotePacket packed;
        packed.rank = hiveGlobalRankId;
        packed.messageType = HIVE_REMOTE_EDT_FAIL_MSG; 
        packed.size = sizeof(packed);
        hiveRemoteSendRequestAsync( requester, (char *)&packed,  sizeof(packed) );
    }
    return false;
}


void hiveServerSetup( struct hiveConfig * config)
{
    //ASYNC Message Deque Init
    hiveLLServerSetup(config);
    hiveLLServerSetRank(config);
    outInit(hiveGlobalRankCount*config->ports);
    #ifdef SEQUENCENUMBERS
    recSeqNumbers = hiveCalloc(sizeof(u64)*hiveGlobalRankCount);
    #endif
}

void hiveServerProcessPacket(struct hiveRemotePacket * packet)
{
    if(packet->messageType!=HIVE_REMOTE_METRIC_UPDATE_MSG ||
       packet->messageType!=HIVE_REMOTE_SHUTDOWN_MSG)
    {
        hiveUpdatePerformanceMetric(hiveNetworkRecieveBW, hiveThread, packet->size, false);
        hiveUpdatePerformanceMetric(hiveFreeBW + packet->messageType, hiveThread, packet->size, false);
        hiveUpdatePacketInfo(packet->size);
    }
#ifdef SEQUENCENUMBERS
    u64 expSeqNumber = __sync_fetch_and_add(&recSeqNumbers[packet->seqRank], 1U);
    if(expSeqNumber != packet->seqNum)
    {
        PRINTF("MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d\n", expSeqNumber, packet->seqNum, packet->rank, packet->messageType);
    }
#endif
    
    switch(packet->messageType)
    {
        case HIVE_REMOTE_EVENT_SATISFY_SLOT_MSG:
        {
            struct hiveRemoteEventSatisfySlotPacket *pack = (struct hiveRemoteEventSatisfySlotPacket *)(packet);
//            PRINTF("Remote Event Satisfy Slot Recieved %lu %u\n", pack->event, pack->slot);
            hiveEventSatisfySlot(pack->event, pack->db, pack->slot);
            break;
        }   
        case HIVE_REMOTE_EDT_SIGNAL_MSG:
        {
            DPRINTF("EDT Signal Recieved\n");
            struct hiveRemoteEdtSignalPacket *pack = (struct hiveRemoteEdtSignalPacket *)(packet);
            hiveSignalEdt( pack->edt, pack->db, pack->slot);
            break;
        }   
        case HIVE_REMOTE_ADD_DEPENDENCE_MSG:
        {
            DPRINTF("Dependence Recieved\n");
            struct hiveRemoteAddDependencePacket *pack = (struct hiveRemoteAddDependencePacket *)(packet);
            hiveAddDependence( pack->source, pack->destination, pack->slot);
            break;
        }   
        case HIVE_REMOTE_DB_REQUEST_MSG:
        {
            struct hiveRemoteDbRequestPacket *pack = (struct hiveRemoteDbRequestPacket *)(packet);
            if(packet->size != sizeof(*pack))
                PRINTF("Error dbpacket insanity\n");
            hiveRemoteDbSend(pack);
            break;
        }
        case HIVE_REMOTE_DB_SEND_MSG:  
        {
            DPRINTF("Remote Db Recieved\n");
            struct hiveRemoteDbSendPacket *pack = (struct hiveRemoteDbSendPacket *)(packet);
            hiveRemoteHandleDbRecieved(pack);
            break;
        }
        case HIVE_REMOTE_INVALIDATE_DB_MSG:
        {
            DPRINTF("DB Invalidate Recieved\n");
            hiveRemoteHandleInvalidateDb( packet );
            break;
        }
        case HIVE_REMOTE_DB_FULL_REQUEST_MSG:
        {
            struct hiveRemoteDbFullRequestPacket *pack = (struct hiveRemoteDbFullRequestPacket *)(packet);
            hiveRemoteDbFullSend(pack);
            break;
        }
        case HIVE_REMOTE_DB_FULL_SEND_MSG:
        {
            struct hiveRemoteDbFullSendPacket * pack = (struct hiveRemoteDbFullSendPacket *)(packet);
            hiveRemoteHandleDbFullRecieved(pack);
            break;
        }
        case HIVE_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG:
        {
            hiveRemoteHandleSendAlreadyLocal(packet);
            break;
        }
        case HIVE_REMOTE_DB_DESTROY_MSG:
        {
            DPRINTF("DB Destroy Recieved\n");
            hiveRemoteHandleDbDestroy( packet );
            break;
        }
        case HIVE_REMOTE_DB_DESTROY_FORWARD_MSG:
        {
            DPRINTF("DB Destroy Forward Recieved\n");
            hiveRemoteHandleDbDestroyForward( packet );
            break;
        }
        case HIVE_REMOTE_DB_CLEAN_FORWARD_MSG:
        {
            DPRINTF("DB Clean Forward Recieved\n");
            hiveRemoteHandleDbCleanForward( packet );
            break;
        }
        case HIVE_REMOTE_DB_UPDATE_GUID_MSG:
        {
            DPRINTF("DB Guid Update Recieved\n");
            hiveRemoteHandleUpdateDbGuid( packet );
            break;
        }
        case HIVE_REMOTE_EDT_MOVE_MSG:
        {
            DPRINTF("EDT Move Recieved\n");
            hiveRemoteHandleEdtMove( packet );
            break;
        }
        case HIVE_REMOTE_DB_MOVE_MSG:
        {
            DPRINTF("DB Move Recieved\n");
            hiveRemoteHandleDbMove( packet );
            break;
        }
        case HIVE_REMOTE_DB_UPDATE_MSG:
        {
            hiveRemoteHandleUpdateDb(packet);
            break;
        }
        case HIVE_REMOTE_EVENT_MOVE_MSG:
        {
            hiveRemoteHandleEventMove(packet);
            break;
        }
        case HIVE_REMOTE_EDT_STEAL_MSG:
        {
            DPRINTF("Remote Steal Request %d\n", packet->rank);
            mugEdts(packet->rank, 0);
            break;
        }
        case HIVE_REMOTE_EDT_FAIL_MSG:
        {
            DPRINTF("Remote Steal Fail\n");
            hiveNodeInfo.stealRequestLock=0U;
            break;
        }
        case HIVE_REMOTE_EDT_RECV_MSG:
        {
            DPRINTF("Remote Handle EDT\n");
            unsigned int remoteStolenEdtCount = handleIncomingEdts( (char *)(packet+1), packet->size- sizeof(*packet) );
            hiveUpdatePerformanceMetric(hiveNetworkRecieveBW, hiveThread, (u64) remoteStolenEdtCount, false);
            hiveNodeInfo.stealRequestLock=0U;
            break;
        }
        case HIVE_REMOTE_SHUTDOWN_MSG:
        {
            DPRINTF("Remote Shutdown Request\n");
            hiveLLServerSyncEndRecv();           
            hiveRuntimeStop();
            break;
        }
        case HIVE_REMOTE_GUID_ROUTE_MSG:
        {
            PRINTF("Guid Route Recieved Deprecated Now\n");
            break;
        }
        case HIVE_REMOTE_METRIC_UPDATE_MSG:
        {
            
            struct hiveRemoteMetricUpdate * pack = (struct hiveRemoteMetricUpdate *) (packet);
            DPRINTF("Metric update Recieved %u -> %d %ld\n", hiveGlobalRankId, pack->type, pack->toAdd);
            hiveHandleRemoteMetricUpdate(pack->type, hiveSystem, pack->toAdd, pack->sub);
            break;    
        }
        case HIVE_ACTIVE_MESSAGE_MSG:
        {
            hiveRemoteHandleActiveMessage(packet);
            break;    
        }
        case HIVE_REMOTE_GET_FROM_DB_MSG:
        {
            hiveRemoteHandleGetFromDb(packet);
            break;
        }
        case HIVE_REMOTE_PUT_IN_DB_MSG:
        {
            hiveRemoteHandlePutInDb(packet);
            break;
        }
        case HIVE_REMOTE_SIGNAL_EDT_WITH_PTR_MSG:
        {
            hiveRemoteHandleSignalEdtWithPtr(packet);
            break;
        }
        case HIVE_REMOTE_SEND_MSG:
        {
            hiveRemoteHandleSend(packet);
            break;
        }
        case HIVE_EPOCH_INIT_MSG:
        {
            hiveRemoteHandleEpochInitSend(packet);
            break;
        }
        case HIVE_EPOCH_REQ_MSG:
        {
            hiveRemoteHandleEpochReq(packet);
            break;
        }
        case HIVE_EPOCH_SEND_MSG:
        {
            hiveRemoteHandleEpochSend(packet);
            break;
        }
        case HIVE_ATOMIC_ADD_ARRAYDB_MSG:
        {
            hiveRemoteHandleAtomicAddInArrayDb(packet);
            break;
        }
        case HIVE_ATOMIC_CAS_ARRAYDB_MSG:
        {
            hiveRemoteHandleAtomicCompareAndSwapInArrayDb(packet);
            break;
        }
        case HIVE_EPOCH_INIT_POOL_MSG:
        {
            hiveRemoteHandleEpochInitPoolSend(packet);
            break;
        }
        case HIVE_EPOCH_DELETE_MSG:
        {
            hiveRemoteHandleEpochDelete(packet);
            break;
        }
        case HIVE_REMOTE_BUFFER_SEND_MSG:
        {
            hiveRemoteHandleBufferSend(packet);
            break;
        }
        case HIVE_REMOTE_DB_MOVE_REQ_MSG:
        {
            hiveDbMoveRequestHandle(packet);
            break;
        }
        default:
        {
            PRINTF("Unknown Packet %d %d %d\n", packet->messageType, packet->size, packet->rank);
            hiveShutdown();
            hiveRuntimeStop();
        }
    }
}


