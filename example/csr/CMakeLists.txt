set(LIBS hive ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

add_executable(csrExample distributedCsr.c csrExample.c)
target_link_libraries(csrExample ${LIBS})
