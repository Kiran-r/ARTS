set(LIBS arts ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

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

add_executable(testArtsSend testArtsSend.c)
target_link_libraries(testArtsSend ${LIBS})

add_executable(testActiveMessageDb testActiveMessageDb.c)
target_link_libraries(testActiveMessageDb ${LIBS})

add_executable(testActiveMessageBuffer testActiveMessageBuffer.c)
target_link_libraries(testActiveMessageBuffer ${LIBS})

add_executable(testCSR testCSR.c)
target_link_libraries(testCSR arts-graph ${LIBS})

#add_executable(testDistribution testDistribution.c)
#target_link_libraries(testDistribution ${LIBS})

add_executable(testArrayDb testArrayDb.c)
target_link_libraries(testArrayDb ${LIBS})

add_executable(testForEach testForEach.c)
target_link_libraries(testForEach ${LIBS})

add_executable(testTerminationDetection testTerminationDetection.c)
target_link_libraries(testTerminationDetection ${LIBS})

add_executable(testTerminationDetection2 testTerminationDetection2.c)
target_link_libraries(testTerminationDetection2 ${LIBS})

add_executable(testPutArrayDbEpoch testPutArrayDbEpoch.c)
target_link_libraries(testPutArrayDbEpoch ${LIBS})

add_executable(testPutArrayDbEpoch2 testPutArrayDbEpoch2.c)
target_link_libraries(testPutArrayDbEpoch2 ${LIBS})

add_executable(testAtomicAddInArrayDb testAtomicAddInArrayDb.c)
target_link_libraries(testAtomicAddInArrayDb ${LIBS})

add_executable(testAtomicCompareAndSwapInArrayDb testAtomicCompareAndSwapInArrayDb.c)
target_link_libraries(testAtomicCompareAndSwapInArrayDb ${LIBS})

add_executable(testDbMove testDbMove.c)
target_link_libraries(testDbMove ${LIBS})

add_executable(testDbOnce testDbOnce.c)
target_link_libraries(testDbOnce ${LIBS})

add_executable(testRecEpoch testRecEpoch.c)
target_link_libraries(testRecEpoch ${LIBS})

add_executable(testEpochPool testEpochPool.c)
target_link_libraries(testEpochPool ${LIBS})

add_executable(testBuffer testBuffer.c)
target_link_libraries(testBuffer ${LIBS})

add_executable(testEvent testEvent.c)
target_link_libraries(testEvent ${LIBS})