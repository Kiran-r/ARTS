set(LIBS arts ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

if(CUDA_FOUND)
    add_library(vecAdd_test_cuda vecAdd.cu)
    set_target_properties(vecAdd_test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(vecAdd_test vecAdd.c)
    target_link_libraries(vecAdd_test vecAdd_test_cuda ${LIBS} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

    add_executable(vecAddNoArts vecAddNoArts.c)
    target_link_libraries(vecAddNoArts vecAdd_test_cuda ${LIBS} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
else()
    add_executable(vecAdd_test mm.c)
    target_link_libraries(vecAdd_test ${LIBS})

    add_executable(vecAddNoArts vecAddNoArts.c)
    target_link_libraries(vecAddNoArts ${LIBS})
endif()
