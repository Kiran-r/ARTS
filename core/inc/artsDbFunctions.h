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
#ifndef ARTSDBFUNCTIONS_H
#define	ARTSDBFUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

void artsDbCreateInternal(artsGuid_t guid, void *addr, uint64_t size, uint64_t packetSize, artsType_t mode);    
void acquireDbs(struct artsEdt * edt);
void releaseDbs(unsigned int depc, artsEdtDep_t * depv);
bool artsAddDbDuplicate(struct artsDb * db, unsigned int rank, struct artsEdt * edt, unsigned int slot, artsType_t mode);
void prepDbs(unsigned int depc, artsEdtDep_t * depv);
void internalPutInDb(void * ptr, artsGuid_t edtGuid, artsGuid_t dbGuid, unsigned int slot, unsigned int offset, unsigned int size, artsGuid_t epochGuid, unsigned int rank);

#ifdef __cplusplus
}
#endif
#endif	/* artsDBFUNCTIONS_H */

