#ifdef MATRIX_MULTIPLICATION_STREAM_H
#define MATRIX_MULTIPLICATION_STREAM_H

// #ifdef USE_GPU
#include <cuda_runtime.h>
// #endif

void vecAddGPU();
void vecAddStream();

#endif // MATRIX_MULTIPLICATION_STREAM_H
