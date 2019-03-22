include (CheckIncludeFiles)
include (CheckSymbolExists)

set(LIBS arts ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

CHECK_INCLUDE_FILES (alloca.h HAVE_ALLOCA_H)
check_symbol_exists (alloca alloca.h HAVE_ALLOCA)

if(HAVE_ALLOCA)
  add_definitions(-DHAVE_ALLOCA)
endif()

add_executable(fib fib.c)
target_link_libraries(fib ${LIBS})

add_executable(fibYield fibYield.c)
target_link_libraries(fibYield ${LIBS})


add_executable(fibContextSwitch fibContextSwitch.c)
target_link_libraries(fibContextSwitch ${LIBS})

add_executable(simpleReduction simpleReduction.c)
target_link_libraries(simpleReduction ${LIBS})

add_executable(bfs bfs.c)
target_link_libraries(bfs arts-graph ${LIBS})

add_executable(graphSearch graphSearch.c)
target_link_libraries(graphSearch arts-graph ${LIBS})

#SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

add_executable(scanStatistics scanStatistics.cpp)
target_link_libraries(scanStatistics arts-graph ${LIBS})

add_executable(triangleCount triangleCount.c)
target_link_libraries(triangleCount arts-graph ${LIBS})

add_executable(triangleCount2 triangleCount2.c)
target_link_libraries(triangleCount2 arts-graph ${LIBS})
