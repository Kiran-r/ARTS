//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
#ifndef ARTSREMOTE_H
#define ARTSREMOTE_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"
#include "artsConfig.h"

void artsRemoteSetMessageTable( struct artsConfig * table );
void artsRemoteStartProcesses( unsigned int argc, char ** argv );
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
void artsRemoteShutdownPing( unsigned int route);
#ifdef __cplusplus
}
#endif

#endif
