set(LIBS hive ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

add_executable(testMain testMain.c)
target_link_libraries(testMain ${LIBS})

add_executable(testRouteTable testRouteTable.c)
target_link_libraries(testRouteTable ${LIBS})

add_executable(testCDAG testCDAG.c)
target_link_libraries(testCDAG ${LIBS})

add_executable(testCDAG2 testCDAG2.c)
target_link_libraries(testCDAG2 ${LIBS})

#add_executable(testOOList testOOList.c)
#target_link_libraries(testOOList ${LIBS})

add_executable(testPinDB testPinDB.c)
target_link_libraries(testPinDB ${LIBS})

add_executable(testGetFromDb testGetFromDb.c)
target_link_libraries(testGetFromDb ${LIBS})

add_executable(testPutInDb testPutInDb.c)
target_link_libraries(testPutInDb ${LIBS})

add_executable(testHiveSend testHiveSend.c)
target_link_libraries(testHiveSend ${LIBS})

add_executable(testActiveMessageDb testActiveMessageDb.c)
target_link_libraries(testActiveMessageDb ${LIBS})

add_executable(testActiveMessageBuffer testActiveMessageBuffer.c)
target_link_libraries(testActiveMessageBuffer ${LIBS})

add_executable(testCSR testCSR.c)
target_link_libraries(testCSR hive-graph ${LIBS})

#add_executable(testDistribution testDistribution.c)
#target_link_libraries(testDistribution ${LIBS})

add_executable(testArrayDb testArrayDb.c)
target_link_libraries(testArrayDb ${LIBS})

add_executable(testForEach testForEach.c)
target_link_libraries(testForEach ${LIBS})

add_executable(testTerminationDetection testTerminationDetection.c)
target_link_libraries(testTerminationDetection ${LIBS})

