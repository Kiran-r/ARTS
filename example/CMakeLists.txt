set(LIBS hive ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

add_executable(fib fib.c)
target_link_libraries(fib ${LIBS})

add_executable(simpleReduction simpleReduction.c)
target_link_libraries(simpleReduction ${LIBS})

#add_subdirectory(csr)
#add_subdirectory(rmat)
