#ifndef ARTSREMOTE_H
#define ARTSREMOTE_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsConfig.h"

void artsRemoteCheckProgress( int rank);
void artsRemoteSetMessageTable( struct artsConfig * table );
void artsRemoteStartProcesses( unsigned int argc, char ** argv );
void artsRemoteCleanup( );
void artsServerSetup(struct artsConfig * config);
void artsRemoteSetupOutgoing();
bool artsRemoteSetupIncoming();
unsigned int artsRemoteGetMyRank();
void artsRemoteShutdown();
bool artsServerTryToRecieve(char ** inBuffer, int * inPacketSize, volatile unsigned int * remoteStealLock);
void artsServerSendStealRequest();
unsigned int artsRemoteSendRequest( int rank, unsigned int queue, char * message, unsigned int length );
unsigned int artsRemoteSendPayloadRequest( int rank, unsigned int queue, char * message, unsigned int length, char * payload, int length2 );

uint8_t artsEventSatisfyNoBlock(artsGuid_t eventGuid, artsGuid_t dataGuid);
unsigned int artsRemoteDivision();
void artsRemoteTryToBecomePrinter();
void artsRemoteTryToClosePrinter();
void artsServerPingPongTestRecieve(char * inBuffer, int inPacketSize);
void artsRemotSetThreadInboundQueues(unsigned int start, unsigned int stop);
void artsRemoteCleanup();
void artsRemoteShutdownPing( unsigned int route);
#ifdef __cplusplus
}
#endif

#endif
