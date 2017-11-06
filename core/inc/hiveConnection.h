#ifndef HIVETCPOVERIDE_H
#if defined (USE_TCP)
#include <sys/poll.h>
#define rrecv recv
#define rsend send
#define rlisten listen
#define rpoll poll
#define rselect select
#define rbind bind
#define rclose close
#define raccept accept
#define rconnect connect
#define rsocket socket
#define rshutdown shutdown
#elif defined (USE_RSOCKETS)
#include <rdma/rsocket.h>
#endif
#endif
