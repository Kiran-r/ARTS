#ifndef HIVEREMOTE_H
#define HIVEREMOTE_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hive.h"
#include "hiveConfig.h"

void hiveRemoteCheckProgress( int rank);
void hiveRemoteSetMessageTable( struct hiveConfig * table );
void hiveRemoteStartProcesses( unsigned int argc, char ** argv );
void hiveRemoteCleanup( );
void hiveServerSetup(struct hiveConfig * config);
void hiveRemoteSetupOutgoing();
bool hiveRemoteSetupIncoming();
unsigned int hiveRemoteGetMyRank();
void hiveRemoteShutdown();
bool hiveServerTryToRecieve(char ** inBuffer, int * inPacketSize, volatile unsigned int * remoteStealLock);
void hiveServerSendStealRequest();
unsigned int hiveRemoteSendRequest( int rank, unsigned int queue, char * message, unsigned int length );
unsigned int hiveRemoteSendPayloadRequest( int rank, unsigned int queue, char * message, unsigned int length, char * payload, int length2 );

u8 hiveEventSatisfyNoBlock(hiveGuid_t eventGuid, hiveGuid_t dataGuid);
unsigned int hiveRemoteDivision();
void hiveRemoteTryToBecomePrinter();
void hiveRemoteTryToClosePrinter();
void hiveServerPingPongTestRecieve(char * inBuffer, int inPacketSize);
void hiveRemotSetThreadInboundQueues(unsigned int start, unsigned int stop);
void hiveRemoteCleanup();
void hiveRemoteShutdownPing( unsigned int route);
#ifdef __cplusplus
}
#endif

#endif
