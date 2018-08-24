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
#ifndef ARTSARRAYDB_H
#define ARTSARRAYDB_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts.h"

unsigned int artsGetSizeArrayDb(artsArrayDb_t * array);
unsigned int getOffsetFromIndex(artsArrayDb_t * array, unsigned int index);
unsigned int getRankFromIndex(artsArrayDb_t * array, unsigned int index);
artsGuid_t getArrayDbGuid(artsArrayDb_t * array);
void internalAtomicAddInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int toAdd, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);
void internalAtomicCompareAndSwapInArrayDb(artsGuid_t dbGuid, unsigned int index, unsigned int oldValue, unsigned int newValue, artsGuid_t edtGuid, unsigned int slot, artsGuid_t epochGuid);

#ifdef __cplusplus
}
#endif
#endif /* ARTSARRAYDB_H */

