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
//Once this *class* works we will put a stream(s) in create a thread local
//stream.  Then we will push stuff!

typedef struct 
{
    volatile unsigned int scheduled;
    cudaStream_t * stream;
} artsGpuStream_t;

void artsInitGpuStream(artsGpuStream_t ** stream)
{
//    cudaStreamCreate ( cudaStream_t* pStream );
}

void artsDestroyGpuStream(artsGpuStream_t * stream)
{
//    cudaStreamSynchronize ( cudaStream_t stream )
//    	cudaStreamDestroy ( cudaStream_t stream )
}

void artsScheduleToStream(artsGpuStream_t * stream, artsGpu_t fnPtr, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
//    For now this should push the following into the stream:
//    1. Copy data from host to device
//    2. Push kernel
//    3. Copy data from device to host
//    4. Call host callback function artsGpuHostWrapUp
    
//    cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
//    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
//    cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
//    cudaLaunchHostFunc ( cudaStream_t stream, cudaHostFn_t fn, void* userData )
}

void artsWaitForStream(artsGpuStream_t * stream)
{
//    cudaStreamSynchronize ( cudaStream_t stream )
}

void artsStreamBusy(artsGpuStream * stream)
{
//    cudaStreamQuery ( cudaStream_t stream )
}