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
#ifndef ARTSTHREADS_H
#define ARTSTHREADS_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "artsConfig.h"

void artsThreadInit(struct artsConfig * config);
void artsThreadMainJoin();
void artsThreadSetOsThreadCount(unsigned int threads);
void artsPthreadAffinity(unsigned int cpuCoreId);
void artsPthreadCheckAffinity(unsigned int threadId, unsigned int cpuCoreId);
int artsCheckAffinity();
#ifdef __cplusplus
}
#endif
#endif
