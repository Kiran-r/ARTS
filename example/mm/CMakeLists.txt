set(LIBS arts ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

if(CUDA_FOUND)
    add_library(mm_test_cuda mm.cu)
    set_target_properties(mm_test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(mm_test mm.c)
    target_link_libraries(mm_test mm_test_cuda ${LIBS} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

    add_executable(mmNoArts mmNoArts.c)
    target_link_libraries(mmNoArts mm_test_cuda ${LIBS} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})
else()
    add_executable(mm_test mm.c)
    target_link_libraries(mm_test ${LIBS})

    add_executable(mmNoArts mmNoArts.c)
    target_link_libraries(mmNoArts ${LIBS})
endif()
