add_subdirectory(distribution)
add_subdirectory(datastructures)

add_library(arts-graph $<TARGET_OBJECTS:distribution> $<TARGET_OBJECTS:datastructures>)
