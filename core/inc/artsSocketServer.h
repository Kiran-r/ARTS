#ifndef ARTSSOCKETSERVER_H
#define ARTSSOCKETSERVER_H
#ifdef __cplusplus
extern "C" {
#endif
unsigned int artsGetNewSocket();
void artsServerSetSocketOptionsSender(unsigned int socket);
void artsServerSetSocketOptionsReciever(unsigned int socket);
void artsPrintSocketAddr(struct sockaddr_in *sock);
unsigned int artsGetSocketListening( struct sockaddr_in * listeningSocket, unsigned int port );
unsigned int artsGetSocketOutgoing( struct sockaddr_in * outgoingSocket, unsigned int port, in_addr_t s_addr );
#ifdef __cplusplus
}
#endif

#endif
