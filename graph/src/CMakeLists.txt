add_subdirectory(distribution)
add_subdirectory(datastructures)

add_library(hive-graph $<TARGET_OBJECTS:distribution> $<TARGET_OBJECTS:datastructures>)
