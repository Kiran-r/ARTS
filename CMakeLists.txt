cmake_minimum_required(VERSION 2.6)
project (hive)

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

    if(CMAKE_COMPILER_IS_GNUCC)
        message("Using GCC")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message("Using ICC")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Debug Build")
        set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O0 -g -DCNC_DEBUG_TRACE")

        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "-std=gnu99 -O0 -g -DCNC_DEBUG_TRACE")
        endif()
    else()
        message("Release Build")
        set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O2 -g -ftree-vectorize -mtune=native")

        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_C_FLAGS "-std=gnu99 -O2 -g -mtune=native")
        endif()
    endif()

endif()

#Find Required Libs-------------------------->

#find_library(RT_LIB rt)
set(RT_LIB "")
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

#Get HWLOC Libs and includes
find_library(HWLOC_LIB hwloc)
find_path(HWLOC_INCLUDE hwloc.h)

if(HWLOC_LIB AND HWLOC_INCLUDE)
    include_directories(${HWLOC_INCLUDE})
    add_definitions(-DHWLOC)
    message("Using Hwloc")
else()
    message("Hwloc not found.")
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


if(COUNT)
    add_definitions(-DCOUNT)
    message("Counting on")
endif()

if(MODELCOUNT)
    add_definitions(-DMODELCOUNT)
    message("Model Counting only")
endif()

if(INSPECTOR)
    add_definitions(-DINSPECTOR)
    message("Inspection on")
endif()

message("Using ${MALLOC} malloc.")

include_directories(common/inc)
include_directories(core/inc)
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(test)
