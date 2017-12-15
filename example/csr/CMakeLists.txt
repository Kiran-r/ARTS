set(LIBS hive ${M_LIB} ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

add_library(distributedCsr SHARED distributedCsr.c)

add_executable(csrExample csrExample.c)
target_link_libraries(csrExample distributedCsr ${LIBS})
