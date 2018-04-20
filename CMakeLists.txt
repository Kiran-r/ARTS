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
        set(CMAKE_C_FLAGS "-Werror -std=gnu99 -O2 -g -ftree-vectorize -mtune=native")

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

find_package(PkgConfig)

#Get HWLOC Libs and includes

##TODO: enable pkgconfig option
if(HWLOC_ROOT)
    #TODO: Find HWLOC version. Assuming version 2 and above used
    if(PKG_CONFIG_FOUND)
        set(ENV{PKG_CONFIG_PATH} "${HWLOC_ROOT}/lib/pkgconfig")
    endif()
    message("Assuming HWLOC Version 2")
    add_definitions(-DHWLOC_V2)

    message("HWLOC PATH:${HWLOC_ROOT}")

    find_path(Hwloc_INCLUDE_DIRS
      NAMES hwloc.h
      PATHS ${HWLOC_ROOT}/include
      #PATH_SUFFIXES
    )
    message("HWLOC INC PATH:${Hwloc_INCLUDE_DIRS}")
    if(Hwloc_INCLUDE_DIRS)
        set(HWLOC_INCLUDE ${Hwloc_INCLUDE_DIRS})
    endif()

    find_library(Hwloc_LIBRARIES
      NAMES hwloc
      PATHS ${HWLOC_ROOT}/lib ${HWLOC_ROOT}/lib64
      #PATHS_SUFFIXES debug
    )
    message("HWLOC LIBRARY PATH:${Hwloc_LIBRARIES}")
    if(Hwloc_LIBRARIES)
        set(HWLOC_LIB ${Hwloc_LIBRARIES})
    endif()

    if(Hwloc_INCLUDE_DIRS AND Hwloc_LIBRARIES)
        include_directories(${Hwloc_INCLUDE_DIRS})
        add_definitions(-DHWLOC)
        message("Using Hwloc!")
    else()
        message("Hwloc not found!")
    endif()

else()
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

include_directories(common/inc)
include_directories(core/inc)
include_directories(graph/inc)
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(graph)
add_subdirectory(test)
add_subdirectory(example)

configure_file(sampleConfigs/hive.cfg example/hive.cfg COPYONLY)
configure_file(sampleConfigs/hive.insp example/hive.insp COPYONLY)
