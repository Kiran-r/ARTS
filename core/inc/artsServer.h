#ifndef ARTSSERVER_H
#define ARTSSERVER_H
#ifdef __cplusplus
extern "C" {
#endif
#include "artsConfig.h"
#include "artsRemoteProtocol.h"

void artsLLServerSetRank( struct artsConfig * config);
void artsLLServerSetup(struct artsConfig * config);
void artsServerProcessPacket(struct artsRemotePacket * packet);
bool artsLLServerSyncEndSend( char * message, unsigned int length );
bool artsLLServerSyncEndRecv();
bool artsServerEnd();
void artsLLServerShutdown();
#ifdef __cplusplus
}
#endif

#endif

