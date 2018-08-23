#include "sys/socket.h"
#include "arpa/inet.h"
#include "string.h"
#include "netinet/tcp.h"
#include "artsConnection.h" 
#include "arts.h"
#include <fcntl.h>
#define DPRINTF(...)

void artsPrintSocketAddr(struct sockaddr_in *sock)
{
    //char crap[255];
    char * addr = inet_ntoa(sock->sin_addr);
    if(addr!=NULL)
        DPRINTF("socket addr %s\n", addr );
}

unsigned int artsGetNewSocket()
{
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    return socketOut;
}

unsigned int artsGetSocketListening( struct sockaddr_in * listeningSocket, unsigned int port  )
{
    memset( (char *)listeningSocket, 0, sizeof(*listeningSocket) );
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    listeningSocket->sin_family = AF_INET;
    listeningSocket->sin_addr.s_addr = htonl(INADDR_ANY);
    listeningSocket->sin_port = htons(port);
    return socketOut;
}


unsigned int artsGetSocketOutgoing( struct sockaddr_in * outgoingSocket, unsigned int port, in_addr_t saddr  )
{
    memset( (char *)outgoingSocket, 0, sizeof(*outgoingSocket) );
    unsigned int socketOut = rsocket(PF_INET, SOCK_STREAM, 0);
    outgoingSocket->sin_family = AF_INET;
    outgoingSocket->sin_addr.s_addr = saddr;
    outgoingSocket->sin_port = htons(port);
    artsPrintSocketAddr( outgoingSocket );
    return socketOut;
}
