set(LIBS arts ${RT_LIB} ${PTHREAD_LIB} ${RDMACM_LIB} ${HWLOC_LIB})

if(NOT USE_GPU)
  set (MM_INCLUDE ${CMAKE_SOURCE_DIR}/example/mm)
  set (MM_C_SOURCES ${CMAKE_SOURCE_DIR}/example/mm/mm.c)
  #include_directories(${MM_INCLUDE})
  add_executable(mm_test ${MM_C_SOURCES})
  #target_include_directories(mm_test PUBLIC ${MM_INCLUDE})
  target_link_libraries(mm_test ${LIBS})
else()
  set (MM_INCLUDE ${CMAKE_SOURCE_DIR}/examples/mm)
  set (MM_C_SOURCES mm.c)
  set (MM_CUDA_SOURCES mm.cu)

  set_source_files_properties(${MM_CUDA_SOURCES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  cuda_add_library(mm_test_cuda ${MM_CUDA_SOURCES})

  target_include_directories(mm_test_cuda PUBLIC ${MM_INCLUDE})
  add_executable(mm_test ${MM_C_SOURCES})

  target_include_directories(mm_test PUBLIC ${MM_INCLUDE})
  add_dependencies(mm_test mm_test_cuda)

  #target_link_libraries(mm_test mm_test_cuda ${LIBS} mm_test_cuda ${CUDA_LIBRARIES})

  if(USE_CUBLAS)
    CUDA_ADD_CUBLAS_TO_TARGET(mm_test)
  endif()
  target_link_libraries(mm_test mm_test_cuda ${LIBS} ${CUDA_LIBRARIES})
#
#  add_executable(mmNoArts mmNoArts.c)
#  target_include_directories(mmNoArts PUBLIC ${MM_INCLUDE})
#
#  add_dependencies(mmNoArts mm_test_cuda)
#  target_link_libraries(mmNoArts mm_test_cuda ${LIBS} ${CUDA_LIBRARIES}) 
endif()
