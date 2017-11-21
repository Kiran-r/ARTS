set(LIBS hive ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

add_executable(testMain testMain.c)
target_link_libraries(testMain ${LIBS})

add_executable(fib fib.c)
target_link_libraries(fib ${LIBS})

add_executable(testActiveMessage testActiveMessage.c)
target_link_libraries(testActiveMessage ${LIBS})

add_executable(testRouteTable testRouteTable.c)
target_link_libraries(testRouteTable ${LIBS})

add_executable(testCDAG testCDAG.c)
target_link_libraries(testCDAG ${LIBS})

add_executable(testCDAG2 testCDAG2.c)
target_link_libraries(testCDAG2 ${LIBS})

add_executable(testOOList testOOList.c)
target_link_libraries(testOOList ${LIBS})