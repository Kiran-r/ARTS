###############################################################################
## This material was prepared as an account of work sponsored by an agency 
## of the United States Government.  Neither the United States Government 
## nor the United States Department of Energy, nor Battelle, nor any of 
## their employees, nor any jurisdiction or organization that has cooperated 
## in the development of these materials, makes any warranty, express or 
## implied, or assumes any legal liability or responsibility for the accuracy, 
## completeness, or usefulness or any information, apparatus, product, 
## software, or process disclosed, or represents that its use would not 
## infringe privately owned rights.
##
## Reference herein to any specific commercial product, process, or service 
## by trade name, trademark, manufacturer, or otherwise does not necessarily 
## constitute or imply its endorsement, recommendation, or favoring by the 
## United States Government or any agency thereof, or Battelle Memorial 
## Institute. The views and opinions of authors expressed herein do not 
## necessarily state or reflect those of the United States Government or 
## any agency thereof.
##
##                      PACIFIC NORTHWEST NATIONAL LABORATORY
##                                  operated by
##                                    BATTELLE
##                                    for the
##                      UNITED STATES DEPARTMENT OF ENERGY
##                         under Contract DE-AC05-76RL01830
##
## Copyright 2019 Battelle Memorial Institute
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    https://www.apache.org/licenses/LICENSE-2.0 
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
###############################################################################

#This is all placeholder stuff to figure out what we really need...

# CHECK CUDA
OPTION(WITH_CUDA "Enable CUDA Support (Default: OFF)" OFF)
if(WITH_CUDA)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    set(USE_GPU ON)
    add_definitions(-DUSE_GPU)
    message("USE_GPU Set ON")
    #if(USE_GPU)
    #else()
    #  message(FATAL_ERROR "USE_GPU Not Set ON")
    #endif()
    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-O2;-DVERBOSE")
    SET(CUDA_HOST_COMPILER gcc)
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    list(APPEND CUDA_NVCC_FLAGS "-O2;-DVERBOSE;-gencode; arch=compute_60,code=sm_60")
  else()
    message(FATAL_ERROR "CUDA library not found.")
  endif()
  if(WITH_CUBLAS)
    set(USE_CUBLAS ON)
    add_definitions(-DUSE_CUBLAS)
    message("USE_CUBLAS Set ON")
  endif()
endif(WITH_CUDA)

set(LIBS hive ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

if(NOT USE_GPU)
  set (MM_INCLUDE ${CMAKE_SOURCE_DIR}/example/mm)
  set (MM_C_SOURCES ${CMAKE_SOURCE_DIR}/example/mm/mm.c)
  #include_directories(${MM_INCLUDE})
  add_executable(mm_test ${MM_C_SOURCES})
  #target_include_directories(mm_test PUBLIC ${MM_INCLUDE})
  target_link_libraries(mm_test ${LIBS})
else()
  set (MM_INCLUDE ${CMAKE_SOURCE_DIR}/examples/mm)
  set (MM_C_SOURCES mm.c)
  set (MM_CUDA_SOURCES mm.cu)

  set_source_files_properties(${MM_CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  cuda_add_library(mm_test_cuda ${MM_CUDA_SOURCES})
  target_include_directories(mm_test_cuda PUBLIC ${MM_INCLUDE})
  add_executable(mm_test ${MM_C_SOURCES})
  target_include_directories(mm_test PUBLIC ${MM_INCLUDE})
  add_dependencies(mm_test mm_test_cuda)
  #target_link_libraries(mm_test mm_test_cuda ${LIBS} mm_test_cuda ${CUDA_LIBRARIES})
  if(USE_CUBLAS)
    CUDA_ADD_CUBLAS_TO_TARGET(mm_test)
  endif()
  target_link_libraries(mm_test mm_test_cuda ${LIBS} ${CUDA_LIBRARIES})
endif()