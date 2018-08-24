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
#ifndef SHADADAPTER_H
#define SHADADAPTER_H

#ifdef __cplusplus
extern "C" {
#endif

artsGuid_t artsEdtCreateShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv);
artsGuid_t artsActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size, artsGuid_t epochGuid);
void artsSynchronousActiveMessageShad(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, void * data, unsigned int size);

void artsIncLockShad();
void artsDecLockShad();
void artsCheckLockShad();
void artsStartIntroShad(unsigned int start);
void artsStopIntroShad();
unsigned int artsGetShadLoopStride();

artsGuid_t artsAllocateLocalBufferShad(void ** buffer, uint32_t * sizeToWrite, artsGuid_t epochGuid);

#ifdef __cplusplus
}
#endif

#endif /* SHADADAPTER_H */

