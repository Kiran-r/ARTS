cmake_minimum_required(VERSION 2.6)
project (hive)

#CMake Configuration 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(glibc)
CHECK_GLIBC_VERSION()
if(GLIBC_VERSION)
    message("GLIBC Version Available: ${GLIBC_VERSION}")
    if((${GLIBC_VERSION} VERSION_EQUAL 2.19) OR (${GLIBC_VERSION} VERSION_GREATER 2.19))
        add_definitions(-DGLIBC_NEW)
    endif()
endif()

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

    if(CMAKE_COMPILER_IS_GNUCC)
        message("Using GCC")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message("Using ICC")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug Build")
        if (CMAKE_C_COMPILER_VERSION VERSION_GREATER 7.0)
          set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O0 -g -DCNC_DEBUG_TRACE -fsanitize=address -fsanitize=signed-integer-overflow")
        else()
          #set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O0 -g -DCNC_DEBUG_TRACE -fsanitize=address")
          set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O0 -g -DCNC_DEBUG_TRACE")
        endif()

        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "-std=gnu99 -O0 -g -DCNC_DEBUG_TRACE")
        endif()
    else()
        message("Release Build")
        set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O2 -g -ftree-vectorize -mtune=native -fPIC")

        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "-std=gnu99 -O2 -g -mtune=native")
        endif()
    endif()

endif()

#Find Required Libs-------------------------->

find_library(PTHREAD_LIB pthread)

if(CONNECTION MATCHES "TCP" OR MIC)
    message("Using TCP")
    add_definitions(-DUSE_TCP)
elseif(CONNECTION MATCHES "MPI")
    message("Using MPI")
    add_definitions(-DUSE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
else()
    find_library(RDMACM_LIB rdmacm )
    find_path(RSOCKET_INCLUDE rdma/rsocket.h)
    if(RDMACM_LIB)
        include_directories(${RSOCKET_INCLUDE})
        message("Using RDMA RSockets")
        add_definitions(-DUSE_RSOCKETS)
    else()
        set(RDMACM_LIB "")
        set(RSOCKET_INCLUDE "")
        set(CONNECTION "TCP")
        message("Using TCP")
        add_definitions(-DUSE_TCP)
    endif()
endif()

find_package(Hwloc)
if(NOT HWLOC_LIB)
    set(HWLOC_LIB "")
endif()

#<-------------------------------------------

#DEFAULT SETTINGS
if(NOT DEFINED DEQUE)
    set(DEQUE "chase_lev")
endif(NOT DEFINED DEQUE)

if(NOT DEFINED MALLOC)
    set(MALLOC "Standard")
elseif(${MALLOC} MATCHES "FOA")
    add_definitions(-DFOA)
endif()

if(APPLE)
    set(RT_LIB "")
else()
    find_library(RT_LIB rt)
endif()

if(COUNT)
    message("Counting on")
    add_definitions(-DCOUNT)
endif()

if(MODELCOUNT)
    message("Model Counting only")
    add_definitions(-DMODELCOUNT)
endif()

if(INSPECTOR)
    message("Inspection on")
    add_definitions(-DINSPECTOR)

endif()

message("Using ${MALLOC} malloc.")

set(CORE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/core/inc)

include_directories(core/inc)
include_directories(graph/inc)
add_subdirectory(core)
add_subdirectory(graph)
add_subdirectory(test)
add_subdirectory(example)

configure_file(sampleConfigs/hive.cfg example/hive.cfg COPYONLY)
configure_file(sampleConfigs/hive.insp example/hive.insp COPYONLY)
