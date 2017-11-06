add_subdirectory(gas)
add_subdirectory(introspection)
add_subdirectory(network)
add_subdirectory(system)
add_subdirectory(runtime)

set(main_src
    main.c
    hiveConfig.c
    hiveUtil.c
)

add_library(hive ${main_src} $<TARGET_OBJECTS:gas> $<TARGET_OBJECTS:network> $<TARGET_OBJECTS:introspection> $<TARGET_OBJECTS:runtime> $<TARGET_OBJECTS:system> $<TARGET_OBJECTS:common>)
