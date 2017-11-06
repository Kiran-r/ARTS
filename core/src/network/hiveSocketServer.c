#include "hive.h"
#include "hiveMalloc.h"
#include "hiveGuid.h"
#include "hiveConfig.h"
#include "hiveGlobals.h"
#include "hiveRuntime.h"
#include "hiveRuntime.h"
#include "hiveRemote.h"
#include "hiveAtomics.h"
#include "hiveDeque.h"
#include "hiveRemoteProtocol.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "sys/types.h"
#include "netinet/in.h"
#include "netdb.h"
#include "arpa/inet.h"
#include "errno.h"
#include "sys/ioctl.h"
#include "net/if.h"
#include "sys/types.h"
#include "hiveServer.h"
#include "hiveSocketServer.h"
#include "hiveRemoteProtocol.h"
#include "hiveConnection.h"
#include <inttypes.h>
#include "hiveCounter.h"
#include "hiveIntrospection.h"
#include "hiveRemoteFunctions.h"
#include "hiveEdtFunctions.h"
#include <unistd.h>
#include <ifaddrs.h>
#include "hiveRouteTable.h"
#include "hiveDebug.h"
//#include <linux/if_packet.h>
//#include <linux/if_arp.h>

//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )
#define DPRINTF( ... )

#ifdef COUNT
extern __thread u64 packetProcStartTime;
extern __thread struct hiveRemotePacket * localPacket;
#define SETPACKET(pk) localPacket=pk
#else
#define SETPACKET(pk)
#endif

struct hiveConfig * hiveGlobalMessageTable;
unsigned int ports;
//SOCKETS!
int * remoteSocketSendList;
volatile unsigned int * volatile  remoteSocketSendLockList;
struct sockaddr_in * remoteServerSendList;
bool * remoteConnectionAlive;

int * localSocketRecieve;
int * remoteSocketRecieveList;
fd_set readSet;
int maxFD;
struct sockaddr_in * remoteServerRecieveList;
struct pollfd * pollIncoming;

#define EDT_MUG_SIZE 32 
#define PACKET_SIZE 4194304
#define INITIAL_OUT_SIZE 80000000 

char * ipList;

int lastRank;

void hiveRemoteCheckProgress(int route)
{
    
}

void hiveRemoteSetMessageTable( struct hiveConfig * table )
{
    hiveGlobalMessageTable = table;
    ports=table->ports;
}

bool hostnameToIp( char * hostName, char* ip)
{
    int j;
    struct hostent * he;
    struct in_addr **addr_list;
    struct addrinfo* result;
    int error = getaddrinfo(hostName, NULL, NULL, &result);
    if(error == 0)
    {
       
        if(result->ai_addr->sa_family == AF_INET)
        {
            struct sockaddr_in * res = (struct sockaddr_in *)result->ai_addr;
            inet_ntop(AF_INET, &res->sin_addr, ip, 100);
        }
        else if(result->ai_addr->sa_family == AF_INET6)
        {
            struct sockaddr_in6 * res = (struct sockaddr_in6 *)result->ai_addr;
            inet_ntop(AF_INET6, &res->sin6_addr, ip, 100);
        }
        freeaddrinfo(result); 
        return true;
    }
    PRINTF("%s\n",gai_strerror(error));

    return false;
}

void hiveRemoteFixNames( char * fix, unsigned int fixLength, bool isItPost, char ** fixMe)
{
    char * oldStr, * newStr;
    int oldStrLength;
    //for(int i=0; i<hiveGlobalMessageTable->tableLength; i++)
    {
        oldStr = *fixMe;//hiveGlobalMessageTable->table[i].ipAddress;
        oldStrLength = strlen(oldStr);

        newStr = hiveMalloc(oldStrLength+fixLength+1); 
        
        if(isItPost)
        {
            strncpy( newStr, oldStr, oldStrLength );
            strncpy( newStr+oldStrLength, fix, fixLength ); 
            *(newStr+fixLength+oldStrLength) = '\0';
            *fixMe = newStr;
            hiveFree(oldStr);
        }
        else
        {
            strncpy( newStr, fix, fixLength ); 
            strncpy( newStr+fixLength, oldStr, oldStrLength );
            *(newStr+fixLength+oldStrLength) = '\0';
            //hiveGlobalMessageTable->table[i].ipAddress = newStr;
            *fixMe = newStr;
            hiveFree(oldStr);
        }
    }

}

void hiveServerFixIbNames(struct hiveConfig * config )
{
    struct hostent * he;
    
    bool found=false;

    char post[5][10] = {"-ib\0", "ib\0", ".ib\0","-ib.ib\0", "\0"};
    char pre[4][10] = {"ib-\0", "ib\0", "ib.\0", "\0"};
   
    int curLength;
    for(int j =0; j<config->tableLength; j++)
    {
    char * testStr = hiveGlobalMessageTable->table[j].ipAddress;
    int testStrLength = strlen(testStr);
    char * stringFixed = hiveMalloc(testStrLength+50);
    struct addrinfo* result;
    bool found = false;
    int i=0, error;
    while(pre[i][0] != '\0' &&!found)
    {
        curLength = strlen(pre[i]);
        
        strncpy( stringFixed, pre[i], curLength ); 
        strncpy( stringFixed+curLength, testStr, testStrLength );
        *(stringFixed+curLength+testStrLength) = '\0';
        
        DPRINTF("%s\n",stringFixed);

        error = getaddrinfo(stringFixed, NULL, NULL, &result);

        if(error == 0)
        {
            hiveRemoteFixNames(pre[i], curLength, false, &hiveGlobalMessageTable->table[j].ipAddress); 
            hiveFree(stringFixed);
            
            freeaddrinfo(result); 
            found = true;
        }

        i++;
        
    }
    
    i=0;
    while(post[i][0] != '\0' && !found)
    {
        curLength = strlen(post[i]);
        
        strncpy( stringFixed, testStr, testStrLength );
        strncpy( stringFixed+testStrLength, post[i], curLength ); 
        *(stringFixed+curLength+testStrLength) = '\0';
        
        DPRINTF("%s\n",stringFixed);
        
        error = getaddrinfo(stringFixed, NULL, NULL, &result);

        if(error == 0)
        {
            hiveRemoteFixNames(post[i], curLength, true, &hiveGlobalMessageTable->table[j].ipAddress); 
            hiveFree(stringFixed);
            
            freeaddrinfo(result); 
            found= true;
        }

        i++;
        
    }
    }

}
bool workAround = false;
void hiveLLServerSetRank( struct hiveConfig * config)
{
}

bool hiveServerSetIP( struct hiveConfig * config )
{
    
    ipList = hiveMalloc(100*sizeof(char ) * config->tableLength);
    
    bool result;

    for(int i =0; i<config->tableLength; i++)
    {
        result = hostnameToIp(config->table[i].ipAddress, ipList+100*i);
        //result = hostnameToIp("www.google.com", ipList+100*i);

        if(!result)
        {
            PRINTF("Cannot get ip address for '%s'\n", config->table[i].ipAddress);
            exit(1);
        }
    }

    int fd;
    struct ifreq ifr;
    char * connection = NULL;

    ifr.ifr_addr.sa_family = AF_INET;
    bool found = false;

    //if(config->netInterface == NULL)
    {
         struct ifaddrs *ifap, *ifa;
         struct sockaddr_in *sa;
         struct sockaddr_in6 *sa6;
         char addr[100];

         getifaddrs (&ifap);
         for (ifa = ifap; ifa && !found; ifa = ifa->ifa_next)
         {
            
            if (ifa->ifa_addr->sa_family==AF_INET)
            {
                sa = (struct sockaddr_in *) ifa->ifa_addr;
                inet_ntop(AF_INET, &sa->sin_addr, addr, 100);
                DPRINTF("Interface: %s\tAddress: %s\n", ifa->ifa_name, addr);
                
                for(int i=0; i< config->tableLength && !found; i++)
                {
                    if(strcmp(addr,ipList+100*i) == 0)
                    {
                        found = true;
                        config->myRank = i;
                        hiveGlobalRankId = i;
                        hiveGlobalRankCount = hiveGlobalMessageTable->tableLength;
                    }
                }
            }
            else if (ifa->ifa_addr->sa_family==AF_INET6)
            {
                sa6 = (struct sockaddr_in6 *) ifa->ifa_addr;
                inet_ntop(AF_INET6, &sa6->sin6_addr, addr, 100);
                DPRINTF("Interface: %s\tAddress: %s\n", ifa->ifa_name, addr);
                
                for(int i=0; i< config->tableLength && !found; i++)
                {
                    if(strcmp(addr,ipList+100*i) == 0)
                    {
                        found = true;
                        config->myRank = i;
                        hiveGlobalRankId = i;
                        hiveGlobalRankCount = hiveGlobalMessageTable->tableLength;
                    }
                }
            
            }
         }
    }
    if(found)
        return true;
    else
        return false;
}

void hiveLLServerSetup(struct hiveConfig * config)
{
    hiveRemoteSetMessageTable(config);
    #if defined (USE_TCP)
    if(config->table && config->ibNames)
        hiveServerFixIbNames(config);
    #else
    if(config->table)
        hiveServerFixIbNames(config);
    #endif

    if(!hiveServerSetIP(config) && config->nodes > 1)
    {
        //PRINTF("[%d]Could not connect to %s\n", hiveGlobalRankId, config->netInterface);
        PRINTF("Could not resolve ip to any device\n");
        exit(1);
    }


}


void hiveLLServerShutdown()
{
    //hiveDebugPrintStack();
    //PRINTF("Here\n");
    int count = hiveGlobalMessageTable->tableLength;
    for(int i=0; i<(count-1)*ports; i++)
    {
        rshutdown(remoteSocketRecieveList[i], SHUT_RDWR);
        //rclose(remoteSocketRecieveList[i]);
    }
    
    for(int i=0; i<count*ports; i++)
    {
        if(i/ports!=hiveGlobalRankId)
        {
            rshutdown(remoteSocketSendList[i], SHUT_RDWR);
//            rclose(remoteSocketSendList[i]);
        }
    }

}

unsigned int hiveRemoteGetMyRank()
{
    DPRINTF("My rank %d\n", hiveGlobalMessageTable->myRank);
    return hiveGlobalMessageTable->myRank;
}

bool set=false; 
static inline bool hiveRemoteConnect( int rank, unsigned int port )
{

    DPRINTF("connecy try %d\n", rank);
    //sleep(10);
    if(!remoteConnectionAlive[rank*ports+port])
    {
        DPRINTF("connecy %d %d\n", rank, remoteSocketSendList[rank*ports+port] );
        hivePrintSocketAddr( &remoteServerSendList[rank*ports+port] );
        int res = rconnect( remoteSocketSendList[rank*ports+port], (struct sockaddr*)(remoteServerSendList+rank*ports+port), sizeof(struct sockaddr_in) );
        if( res < 0 )
        {
            //if(hiveGlobalRankId==0)
            //    hiveDebugGenerateSegFault();
            void * ptrCrap;
            DPRINTF("%d error %s %d %p %d %s\n", rank, strerror(errno), errno, ptrCrap, remoteSocketSendList[rank], hiveGlobalMessageTable->table[rank].ipAddress);
            DPRINTF("[%d]Connect Failed to rank %d %d\n", hiveGlobalRankId, rank, res);
            
            remoteConnectionAlive[rank] = false;
            
            rclose( remoteSocketSendList[rank*ports+port] );
            remoteSocketSendList[rank*ports+port] = hiveGetNewSocket();

            while( rconnect( remoteSocketSendList[rank*ports+port], (struct sockaddr*)(remoteServerSendList+rank*ports+port), sizeof(struct sockaddr_in) ) <0 )
            {
                rclose( remoteSocketSendList[rank*ports+port] );
                remoteSocketSendList[rank*ports+port] = hiveGetNewSocket();
            
            }

            DPRINTF("Connect now succedded to rank %d %d\n", rank, res);
            remoteConnectionAlive[rank*ports+port] = true;

            return true;
        }

        remoteConnectionAlive[rank*ports+port] = true;
    }

    return true;
}

bool hiveLLServerSyncEndSend( char * message, unsigned int length )
{
    return false;
}

bool hiveLLServerSyncEndRecv()
{
    return false;
}

unsigned int hiveRemoteSendRequest( int rank, unsigned int queue, char * message, unsigned int length  )
{
    int res=0;
    int port = queue % ports;
    if(hiveRemoteConnect(rank, port))
    {
        int backOff=1;
        /*while( hiveAtomicSwap( remoteSocketSendLockList+rank, 1U  ) != 0U )
        {
            usleep(backOff++);
        }*/
        DPRINTF("sent %d\n", rank);

        int total=0;
        int lastLength = length;
        
#ifdef COUNT
        //struct hiveRemotePacket * pk = (void *)message;
        //if(!pk->timeStamp)
        //    pk->timeStamp = hiveExtGetTimeStamp();
#endif
        
        

        while(length!=0 && res >= 0)
        {
            res = rsend( remoteSocketSendList[rank*ports+port], message+total, length, MSG_DONTWAIT );
            if(res >= 0)
            {
                lastLength = length;
                total+=res;
                length-=res;
            }
        }

        if( res < 0 )
        {
            //PRINTF("Fail Send\n");
            if (errno != EAGAIN)
            {
                struct hiveRemotePacket * pk = (void *)message;
                PRINTF("hiveRemoteSendRequest %u Socket appears to be closed to rank %d:  %s\n", pk->messageType, rank, strerror(errno));
                //exit(1);
                hiveRuntimeStop();
                return -1;
            }
        }


        //remoteSocketSendLockList[rank]=0U;
        
    }
    return length;

}

unsigned int hiveRemoteSendPayloadRequest( int rank, unsigned int queue, char * message, unsigned int length, char * payload, int length2 )
{
    int res=0;
    int lengthTemp = length2;
    int port = queue % ports;
    if(hiveRemoteConnect(rank, port))
    {
        DPRINTF("sent %d\n", rank);
        int backOff=1;
        /*while( hiveAtomicSwap( remoteSocketSendLockList+rank, 1U  ) != 0U )
        {
            usleep(backOff++);
        }*/

        int total=0;
#ifdef COUNT
        //struct hiveRemotePacket * pk = (void *)message;
        //if(!pk->timeStamp)
        //    pk->timeStamp = hiveExtGetTimeStamp();
#endif
        while(length!=0 && res >= 0)
        {
            res = rsend( remoteSocketSendList[rank*ports+port], message+total, length, MSG_DONTWAIT );
            if(res >= 0)
            {
                total+=res;
                length-=res;
            }
        }

        if( res < 0 )
        {
            //PRINTF("Fail Send payload 1 %d \n", length2+length );
            if (errno != EAGAIN)
            {
                //PRINTF("error %s\n", strerror(errno));
                //PRINTF("Send Failed to rank %d\n", rank);
                struct hiveRemotePacket * pk = (void *)message;
                PRINTF("hiveRemoteSendPayloadRequest %u 1 Socket appears to be closed to rank %d:  %s\n", pk->messageType, rank, strerror(errno));
                hiveRuntimeStop();
                return -1;
                //exit(1);
            }
            //remoteSocketSendLockList[rank]=0U;
            return length2+length;
        }

        total = 0;
        res = 0;

        while(length2!=0 && res >= 0)
        {
            res = rsend( remoteSocketSendList[rank*ports+port], payload+total, length2, MSG_DONTWAIT );
            if(res >= 0 )
            {
                total+=res;
                length2-=res;
            }
        }

        if( res < 0 )
        {
            //PRINTF("Fail Send payload 2 %p %d %d %d\n", message, length, length2, lengthTemp);
            if (errno != EAGAIN)
            {
                //PRINTF("error %s\n", strerror(errno));
                //PRINTF("Send Failed to rank %d\n", rank);
                struct hiveRemotePacket * pk = (void *)message;
                PRINTF("hiveRemoteSendPayloadRequest %u 2 Socket appears to be closed to rank %d:  %s\n", pk->messageType, rank, strerror(errno));
                hiveRuntimeStop();
                return -1;
                //exit(1);
            }

            //remoteSocketSendLockList[rank]=0U;
            return length2;
        }
        //remoteSocketSendLockList[rank]=0U;
    }

    return length2+length;
}

bool hiveRemoteSetupIncoming()
{
    //PRINTF("%d\n", FD_SETSIZE);
    int i, j, k, pos;
    int inPort=hiveGlobalMessageTable->port;
    socklen_t sLength = sizeof(struct sockaddr);
    int count = (hiveGlobalMessageTable->tableLength-1);


    remoteSocketRecieveList = hiveMalloc( sizeof(int)*(count+1)*ports );
    remoteServerRecieveList = hiveCalloc( sizeof(struct sockaddr_in)*(count+1)*ports );
    pollIncoming = hiveMalloc( sizeof(struct pollfd)*(count+1)*ports );
    
    struct sockaddr_in test;

    struct sockaddr_in * localServerAddr = hiveCalloc(ports*sizeof(struct sockaddr_in));
    localSocketRecieve = hiveCalloc(ports*sizeof(int));

    int iSetOption;
    for(i=0; i<hiveGlobalMessageTable->ports; i++)
    {
        localSocketRecieve[i] = hiveGetSocketListening( &localServerAddr[i], inPort+i);

        iSetOption=1;
        setsockopt(localSocketRecieve[i], SOL_SOCKET, SO_REUSEADDR, (char*)&iSetOption, sizeof(iSetOption));
        
        int res = rbind(localSocketRecieve[i], (struct sockaddr *)&localServerAddr[i], sizeof(localServerAddr[i]) );
        
        if(res < 0)
        {
            PRINTF("Bind Failed\n");
            PRINTF("error %s\n", strerror(errno));
            return false;
        }
        
        res = rlisten( localSocketRecieve[i], 2*count);
        
        if(res < 0)
        {
            PRINTF("Listening Failed\n");
            PRINTF("error %s\n", strerror(errno));
            return false;
        }
    }
    
    FD_ZERO(&readSet);
    for( i=0; i < hiveGlobalMessageTable->tableLength; i++ )
    {
        DPRINTF("%d %d\n", hiveGlobalMessageTable->myRank, hiveGlobalMessageTable->table[i].rank);
        if(hiveGlobalMessageTable->myRank == hiveGlobalMessageTable->table[i].rank)
        {
            set = true;
            DPRINTF("Receive go %d\n",i);
            for( j=0; j < count; j++ )
            {
                for(int z=0; z < ports; z++ )
                {
                    DPRINTF("%d\n", j);
                    sLength = sizeof(struct sockaddr_in);
                    //remoteSocketRecieveList[j] = raccept(localSocketRecieve, (struct sockaddr *)&remoteServerRecieveList[j], &sLength );
                    remoteSocketRecieveList[z+j*ports] = raccept(localSocketRecieve[z], (struct sockaddr *)&test, &sLength );
                    
                    if(remoteSocketRecieveList[z+j*ports] < 0 )
                    {
                        int retry =0;
                        PRINTF("Accept Failed\n");
                        PRINTF("error %s\n", strerror(errno));
                        int retryLimit = 3;
                        while(remoteSocketRecieveList[z+j*ports] < 0 )
                        {
                            PRINTF("Retrying %d more times\n", retryLimit-retry);
                            if(retry == retryLimit)
                            {
                               exit(1); 
                            }
                            remoteSocketRecieveList[z+j*ports] = raccept(localSocketRecieve[z], (struct sockaddr *)&test, &sLength );
                            retry++;
                            if(remoteSocketRecieveList[z+j*ports] < 0 )
                            {
                                PRINTF("Accept Failed\n");
                                PRINTF("error %s\n", strerror(errno));
                            }
                        }
                        
                    }
                    //FD_SET(remoteSocketRecieveList[j] , &readSet  );
                    pollIncoming[z+j*ports].fd = remoteSocketRecieveList[z+j*ports];
                    pollIncoming[z+j*ports].events = POLLIN;
                }
            }
        }
        else
        {
            DPRINTF("Connect go %d\n",i);
            for(int z=0; z < ports; z++ )
            {
                if(!hiveRemoteConnect( i, z ))
                {
                    PRINTF("Could not create initial connection\n");
                    return false;
                }
            }
        }
    }
    
    return true;
}

void hiveRemoteSetupOutgoing()
{
    int i, j, k, outPort=hiveGlobalMessageTable->port;
    struct sockaddr_in serverAddress, clientAddress;
    int count = hiveGlobalMessageTable->tableLength;
    struct hostent * he;
    struct in_addr **addr_list;
    char ip[100];
    int pos;

    remoteSocketSendList = hiveMalloc( sizeof(int)*count*ports );
    remoteSocketSendLockList = hiveCalloc( sizeof(int)*count*ports );
    remoteServerSendList = hiveCalloc( sizeof(struct sockaddr_in)*count*ports );
    remoteConnectionAlive = hiveCalloc( sizeof(bool)*count*ports );

    for(i=0; i< count; i++ )
    {
        for(j=0; j< ports; j++ )
            remoteSocketSendList[i*ports+j] = hiveGetSocketOutgoing( remoteServerSendList+i*ports+j, outPort+j, inet_addr(ipList+100*i) );
    }
}
void hiveRemoteCleanup()
{
    
}


static __thread unsigned int threadStart;
static __thread unsigned int threadStop;
static __thread char ** bypassBuf;
static __thread unsigned int * bypassPacketSize;
static __thread unsigned int * reRecieveRes;
static __thread void ** reRecievePacket;
static __thread bool * maxIncoming;
static __thread bool maxOutWorking;

static __thread unsigned int byteRecieveCount=0;
static __thread u64 lastRecieve;
static __thread double maxBandwidth=0;

void hiveRemotSetThreadInboundQueues(unsigned int start, unsigned int stop)
{
    threadStart = start;
    threadStop = stop;
    //MASTER_PRINTF("%d %d\n", start, stop);
    unsigned int size = stop - start;
    bypassBuf = hiveMalloc( sizeof(char*) * size );
    bypassPacketSize = hiveMalloc( sizeof(unsigned int) * size );
    reRecieveRes = hiveCalloc( sizeof(int) * size );
    reRecievePacket = hiveCalloc( sizeof(void *) * size );
    maxIncoming = hiveCalloc( sizeof(bool) * size );
    for(int i=0; i< size; i++)
    {
        bypassBuf[i] = hiveMalloc( PACKET_SIZE );
        bypassPacketSize[i] = PACKET_SIZE;
    }
}

bool maxOutBuffs(unsigned int ignore)
{
    int timeOut=1, res, res2;
    struct hiveRemotePacket * packet;
    //PRINTF("MAX\n");
    res =rpoll(pollIncoming+threadStart, threadStop-threadStart, timeOut );
    unsigned int pos;
    
    if(res == -1)
    {
        hiveShutdown();
        hiveRuntimeStop();
    }
    if(res>0) 
    {
        //PRINTF("MAX LOOP\n");
        timeOut=1;
        for(int i=threadStart; i<threadStop; i++)
        {
            pos = i-threadStart;
            if( i!= ignore && pollIncoming[i].revents & POLLIN )
            {
                maxOutWorking = true;
                if(reRecieveRes[pos] == 0)
                {
                    packet = (struct hiveRemotePacket *)bypassBuf[pos];
                    res = rrecv( remoteSocketRecieveList[i], bypassBuf[pos], bypassPacketSize[pos], 0 );
                }
                else
                {
                    //packet = reRecievePacket[pos];
                    packet = (struct hiveRemotePacket *)bypassBuf[pos];
                    res = reRecieveRes[pos];
                    reRecieveRes[pos] = 0;
                    //if(packet->size > 5000000)
                    //    hiveDebugGenerateSegFault();
                    //PRINTF("Here res %p %d %d\n", packet, res, pos);
                }
                //spaceLeft = bypassPacketSize[pos];
                if( res > 0 )
                {
                    DPRINTF("gg %d %d\n", res, packet->rank);
                    //spaceLeft-=res;
                    while(res < bypassPacketSize[pos] )
                    {
                        DPRINTF("POS Buffffff %d %d\n", res, packet->size);
                        if(bypassBuf[pos]!=(char*)packet )
                        {
                            DPRINTF("memmove\n");
                            memmove(bypassBuf[pos], packet, res);
                            packet = (struct hiveRemotePacket *)bypassBuf[pos];
                            //spaceLeft = bypassPacketSize[pos];
                        }
                        res2 = rrecv( remoteSocketRecieveList[i], bypassBuf[pos]+res, bypassPacketSize[pos ]-res, MSG_DONTWAIT );
                        
                        DPRINTF("res %d %d\n", res, res2);
                        if(res2 < 0 )
                        {
                            if (errno != EAGAIN)
                            {
                                PRINTF("Error on recv return 0 %d %d\n", errno, EAGAIN);
                                PRINTF("error %s\n", strerror(errno));
                                hiveShutdown();
                                hiveRuntimeStop();
                            }
                            
                            reRecieveRes[pos] = res;
                            maxIncoming[pos] = true;
                            //PRINTF("Here\n");
                            //reRecievePacket[pos] = packet;
                            break;
                        }
                        //spaceLeft-=res2;
                        res+=res2;
                    }
                    maxIncoming[pos] = true;
                    reRecieveRes[pos] = res;
                }
                else if( res == -1  )
                {
                    PRINTF("Error on recv socket return 0\n");
                    PRINTF("error %s\n", strerror(errno));
                    hiveShutdown();
                    hiveRuntimeStop();
                    return false;
                }
                else if(res == 0)
                {
                    //PRINTF("Hmm socket close?\n");
                    hiveShutdown();
                    hiveRuntimeStop();
                    return false;
                }
            }
        }
    }
    return true;
}
bool hiveServerTryToRecieve(char ** inBuffer, int * inPacketSize, volatile unsigned int * remoteStealLock)
{
    int i, res, res2, stealHandlerThread=0;
    struct hiveRemotePacket * packet;
    int count = hiveGlobalMessageTable->tableLength-1;
    fd_set tempSet;
    int timeOut=300000;
    //int timeOut=300000;
    struct timeval selTimeout;
    unsigned int pos;
    res =rpoll(pollIncoming+threadStart, threadStop-threadStart, timeOut );
    
    if(res == -1)
    {
        hiveShutdown();
        hiveRuntimeStop();
    }

    unsigned int spaceLeft; 
    bool packetIncomingOnASocket=false;
    bool gotoNext = false;
    if(res>0) 
    {
        //PRINTF("POLL\n");
        timeOut=1;
        maxOutWorking = true;
        while(maxOutWorking)
        {
            maxOutWorking = false;
        for(i=threadStart; i<threadStop; i++)
        {
            pos = i-threadStart;
            gotoNext = false;
            //if( pollIncoming[i].revents & POLLIN )
            //if(!maxOutBuffs(-1))
            //    return false;
            //if( maxIncoming[pos] )
            if( pollIncoming[i].revents & POLLIN )
            {
                //PRINTF("Here2\n");
                maxIncoming[pos] = false;
                HIVECOUNTERTIMERSTART(pktReceive);
                if(reRecieveRes[pos] == 0)
                {
                    //PRINTF("Here3a\n");
                    packet = (struct hiveRemotePacket *)bypassBuf[pos];
                    res = rrecv( remoteSocketRecieveList[i], bypassBuf[pos], bypassPacketSize[pos], 0 );
                }
                else
                {
                    //packet = reRecievePacket[pos];
                    packet = (struct hiveRemotePacket *)bypassBuf[pos];
                    res = reRecieveRes[pos];
                    reRecieveRes[pos] = 0;
                    //PRINTF("Here3\n");
                    //if(packet->size > 5000000)
                    //    hiveDebugGenerateSegFault();
                    //PRINTF("Here res %p %d %d\n", packet, res, pos);
                }
                //spaceLeft = bypassPacketSize[pos];
                if( res > 0 )
                {
                    packetIncomingOnASocket=true;
                    DPRINTF("gg %d %d\n", res, packet->rank);
                    //spaceLeft-=res;
                    while(res>0)
                    {
                        //if(!maxOutBuffs(i))
                        //    return false;
                        while(res < sizeof (struct hiveRemotePacket) )
                        {
                            //PRINTF("Here4\n");
                            DPRINTF("POS Buffffff %d %d\n", res, packet->size);
                            if(bypassBuf[pos]!=(char*)packet )
                            {
                                DPRINTF("memmove\n");
                                memmove(bypassBuf[pos], packet, res);
                                packet = (struct hiveRemotePacket *)bypassBuf[pos];
                                //spaceLeft = bypassPacketSize[pos];
                            }
                            res2 = rrecv( remoteSocketRecieveList[i], bypassBuf[pos]+res, bypassPacketSize[pos ]-res, 0 );
                            
                            DPRINTF("res %d %d\n", res, res2);
                            if(res2 < 0 )
                            {
                                if (errno != EAGAIN)
                                {
                                    PRINTF("Error on recv return 0 %d %d\n", errno, EAGAIN);
                                    PRINTF("error %s\n", strerror(errno));
                                    hiveShutdown();
                                    hiveRuntimeStop();
                                }
                                
                                reRecieveRes[pos] = res;
                                //reRecievePacket[pos] = packet;
                                gotoNext = true;
                                break;
                                //return false;
                            }
                            //spaceLeft-=res2;
                            res+=res2;
                        }
                        if(gotoNext)
                            break;

                        DPRINTF("gg2 %d %d %d %d\n", res, packet->rank, packet->size, packet->messageType);

                        if(bypassPacketSize[pos] < packet->size)
                        {
                            //PRINTF("Here5\n");
                            char * nextBuf = hiveMalloc( packet->size*4 );


                            memcpy(nextBuf, bypassBuf[pos], bypassPacketSize[pos] );

                            hiveFree(bypassBuf[pos]);

                            packet = (struct hiveRemotePacket * )(nextBuf + ( ((char *)packet) - ((char *)bypassBuf[pos])));

                            //*inBuffer = buf = nextBuf;
                            bypassBuf[pos] = nextBuf;
                            
                            //(*inPacketSize) = packetSize = packet->size;
                            bypassPacketSize[pos] = packet->size*4;
                            //spaceLeft = bypassPacketSize[pos];
                        }

                        while( res<packet->size )
                        {
                            //PRINTF("Here6\n");
                            DPRINTF("POS Buffffff a %d %d\n", res, packet->size);
                            //spaceLeft = (bypassPacketSize[pos] - (((char *)packet) - bypassBuf[pos])) - res;
                            //PRINTF("%d %d %d %d\n", spaceLeft, ((char *)packet) - bypassBuf[pos], res, packet->size);
                            //if(bypassBuf[pos]!=(char*)packet && (packet->size-res) > spaceLeft )
                            if(bypassBuf[pos]!=(char*)packet  )
                            {
                                DPRINTF("memmove fix\n");
                                memmove(bypassBuf[pos], packet, res);
                                packet = (struct hiveRemotePacket *)bypassBuf[pos];
                                //spaceLeft = bypassPacketSize[pos];
                            }
                            res2 = rrecv( remoteSocketRecieveList[i], bypassBuf[pos]+res, bypassPacketSize[pos]-res, 0);
                            //res2 = rrecv( remoteSocketRecieveList[i], bypassBuf[pos]+res, bypassPacketSize[pos]-res, MSG_WAITALL );
                            //res2 = rrecv( remoteSocketRecieveList[i], bypassBuf[pos]+res, packet->size-res, MSG_WAITALL );
                            //res2 = rrecv( remoteSocketRecieveList[i], ((char *)packet)+res, spaceLeft, 0 );
                            if(res2 < 0 )
                            {
                                if (errno != EAGAIN)
                                {
                                    PRINTF("Error on recv return 0 %d %d\n", errno, EAGAIN);
                                    PRINTF("error %s\n", strerror(errno));
                                    hiveShutdown();
                                    hiveRuntimeStop();
                                }
                                
                                //PRINTF("Here %p %d %d\n", packet, res, pos);

                                reRecieveRes[pos] = res;
                                //reRecievePacket[pos] = packet;
                                gotoNext = true;
                                break;
                                //return false;
                            }
                            //spaceLeft-=res2;
                            res+=res2;
                            DPRINTF("res %d %d\n", res, res2);
                        }
                        if(gotoNext)
                            break;
                        
                        hiveServerProcessPacket(packet);

                        res-=packet->size;
                        //PRINTF("Here 8 %d\n", res);
                        DPRINTF("PACKET move %d %d\n", res, packet->size );
                        
                        packet = (struct hiveRemotePacket *)( ((char *)packet) + packet->size ); 
                        //memmove(bypassBuf[pos], packet, res);
                        //packet = (struct hiveRemotePacket *)bypassBuf[pos];
                        //reRecieveRes[pos] = res;
                        //break;
                        HIVECOUNTERTIMERENDINCREMENT(pktProc);
#ifdef COUNT
                        localPacket = NULL;
                        packetProcStartTime = 0;                                                        
#endif
                    }
                }
                else if( res == -1  )
                {
                    PRINTF("Error on recv socket return 0\n");
                    PRINTF("error %s\n", strerror(errno));
                    hiveShutdown();
                    hiveRuntimeStop();
                    return false;
                }
                else if(res == 0)
                {
                    //PRINTF("Hmm socket close?\n");
                    hiveShutdown();
                    hiveRuntimeStop();
                    return false;
                }
            }
        }
        }
        return packetIncomingOnASocket;
    }
    return false;
}

void hiveServerPingPongTestRecieve(char * inBuffer, int inPacketSize)
{
    int packetSize = inPacketSize;
    char * buf = inBuffer;
    int i, res, res2, stealHandlerThread=0, pos;
    struct hiveRemotePacket * packet = (struct hiveRemotePacket *)buf;
    int count = hiveGlobalMessageTable->tableLength-1;
    fd_set tempSet;
    int timeOut=100;
    struct timeval selTimeout;
    tempSet = readSet;
    selTimeout.tv_sec = 10;
    selTimeout.tv_usec = timeOut;
    bool recieved = false;

    while(!recieved)
    {
        res =rpoll(pollIncoming, count, timeOut );
        timeOut=1;
        //if(res)
        for(i=0; i<count; i++)
        {
            if( pollIncoming[i].revents & POLLIN )
            {
                packet = (struct hiveRemotePacket *)buf;
                res = rrecv( remoteSocketRecieveList[i], buf, packetSize, 0 );
                if( res > 0 )
                {
                    while(res>0)
                    {
                        while(res < sizeof (struct hiveRemotePacket) )
                        {
                            if(buf!=(char*)packet)
                            {
                                memmove(buf, packet, res);
                                packet = (struct hiveRemotePacket *)buf;
                            }
                            res2 = rrecv( remoteSocketRecieveList[i], buf+res, packetSize-res, 0 );
                            res+=res2;
                            if(res2 == -1)
                            {
                                PRINTF("Error on recv return 0\n");
                                PRINTF("error %s\n", strerror(errno));
                                hiveShutdown();
                                return;
                            }
                        }

                        //PRINTF("Here\n");
                        while( res<packet->size )
                        {
                            //PRINTF("Here %d %d\n", res, packet->size);
                        
                            if(buf!=(char*)packet)
                            {
                                memmove(buf, packet, res);
                                packet = (struct hiveRemotePacket *)buf;
                            }
                            res2 = rrecv( remoteSocketRecieveList[i], buf+res, packetSize-res, 0 );
                            res+=res2;
                            if(res2 == -1)
                            {
                                PRINTF("Error on recv return 0\n");
                                PRINTF("error %s\n", strerror(errno));
                                hiveShutdown();
                                return;
                            }
                        }
                        if( packet->messageType == HIVE_REMOTE_PINGPONG_TEST_MSG)
                        {
                            recieved = true;
                            HIVECOUNTERTIMERENDINCREMENT(pktReceive);
                            hiveUpdatePerformanceMetric(hiveNetworkRecieveBW, hiveThread, packet->size, false);
                            hiveUpdatePerformanceMetric(hiveFreeBW + packet->messageType, hiveThread, 1, false);
                            hiveUpdatePacketInfo(packet->size);
                            //PRINTF("Recv Packet %d %d\n", res, packet->size);
                        }
                        else
                        {
                            PRINTF("Shit Packet %d %d %d\n", packet->messageType, packet->size, packet->rank);
                        }
                        res-=packet->size;
                        packet = (struct hiveRemotePacket *)( ((char *)packet) + packet->size ); 
                    }
                }
                else if( res == -1  )
                {
                    PRINTF("Error on recv socket return 0\n");
                    PRINTF("error %s\n", strerror(errno));
                    hiveShutdown();
                    return;
                }
                else if(res == 0)
                {
                    PRINTF("Hmm socket close?\n");
                    hiveShutdown();
                    return;
                }
            }
        }
    }
}
