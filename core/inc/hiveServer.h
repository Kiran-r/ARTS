#ifndef HIVESERVER_H
#define HIVESERVER_H
#ifdef __cplusplus
extern "C" {
#endif
#include "hiveConfig.h"
#include "hiveRemoteProtocol.h"

void hiveLLServerSetRank( struct hiveConfig * config);
void hiveLLServerSetup(struct hiveConfig * config);
void hiveServerProcessPacket(struct hiveRemotePacket * packet);
bool hiveLLServerSyncEndSend( char * message, unsigned int length );
bool hiveLLServerSyncEndRecv();
bool hiveServerEnd();
void hiveLLServerShutdown();
#ifdef __cplusplus
}
#endif

#endif

