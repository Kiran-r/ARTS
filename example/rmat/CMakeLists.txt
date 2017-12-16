add_library(rmat rmat.c xalloc.c splittable_mrg.c prng.c graph_generator.c mrg_transitions.c)
target_link_libraries(rmat rt m)

add_library(seq-csr seq-csr.c)

add_executable(rmat_generator driver.c)
target_link_libraries(rmat_generator rmat seq-csr)