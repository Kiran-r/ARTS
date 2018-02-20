include (CheckIncludeFiles)
include (CheckSymbolExists)

set(LIBS hive ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

CHECK_INCLUDE_FILES (alloca.h HAVE_ALLOCA_H)
check_symbol_exists (alloca alloca.h HAVE_ALLOCA)

if(HAVE_ALLOCA)
  add_definitions(-DHAVE_ALLOCA)
endif()

add_executable(fib fib.c)
target_link_libraries(fib ${LIBS})

add_executable(simpleReduction simpleReduction.c)
target_link_libraries(simpleReduction ${LIBS})

add_executable(bfs bfs.c)
target_link_libraries(bfs hive-graph ${LIBS})

add_executable(graph_search graph_search.c)
target_link_libraries(graph_search hive-graph ${LIBS})

add_subdirectory(csr)
add_subdirectory(rmat)
