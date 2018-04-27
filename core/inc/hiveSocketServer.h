#ifndef HIVESOCKETSERVER_H
#define HIVESOCKETSERVER_H
#ifdef __cplusplus
extern "C" {
#endif
unsigned int hiveGetNewSocket();
void hiveServerSetSocketOptionsSender(unsigned int socket);
void hiveServerSetSocketOptionsReciever(unsigned int socket);
void hivePrintSocketAddr(struct sockaddr_in *sock);
unsigned int hiveGetSocketListening( struct sockaddr_in * listeningSocket, unsigned int port );
unsigned int hiveGetSocketOutgoing( struct sockaddr_in * outgoingSocket, unsigned int port, in_addr_t s_addr );
#ifdef __cplusplus
}
#endif

#endif
