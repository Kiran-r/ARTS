/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,* 
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

//Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
//and https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
//Once this *class* works we will put a stream(s) in create a thread local
//stream.  Then we will push stuff!
#include "artsGpuRuntime.h"
#include "artsGpuStream.h"
#include "artsEdtFunctions.h"
#include "artsGlobals.h"

void * artsCudaMallocHost(unsigned int size)
{
    void * ptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

void artsCudaFreeHost(void * ptr)
{
    cudaFreeHost(&ptr);
}

artsGuid_t artsEdtCreateGpuDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid, bool hasDepv)
{
//    ARTSEDTCOUNTERTIMERSTART(edtCreateCounter);
    unsigned int depSpace = (hasDepv) ? depc * sizeof(artsEdtDep_t) : 0;
    unsigned int edtSpace = sizeof(struct artsGpuEdt) + paramc * sizeof(uint64_t) + depSpace;
    
    struct artsGpuEdt * edt = (struct artsGpuEdt *) artsMalloc(edtSpace);
    edt->grid = grid;
    edt->block = block;
    edt->endGuid = endGuid;
    edt->slot = slot;
    edt->dataGuid = dataGuid;
    artsGuid_t guid = NULL_GUID;
    artsEdtCreateInternal((artsEdt*) edt, ARTS_GPU_EDT, &guid, route, artsThreadInfo.clusterId, edtSpace, NULL_GUID, funcPtr, paramc, paramv, depc, true, NULL_GUID, hasDepv);
//    ARTSEDTCOUNTERTIMERENDINCREMENT(edtCreateCounter);
    return guid;
}

artsGuid_t artsEdtCreateGpu(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid)
{
    return artsEdtCreateGpuDep(funcPtr, route, paramc, paramv, depc, grid, block, endGuid, slot, dataGuid, true);
}