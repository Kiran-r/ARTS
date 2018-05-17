add_subdirectory(common)
add_subdirectory(gas)
add_subdirectory(introspection)
add_subdirectory(network)
add_subdirectory(system)
add_subdirectory(runtime)

set(main_inc
    ${CORE_INCLUDE_DIR}/hiveRT.h
    ${CORE_INCLUDE_DIR}/hive.h
    ${CORE_INCLUDE_DIR}/hiveGlobals.h
    ${CORE_INCLUDE_DIR}/hiveConfig.h
    ${CORE_INCLUDE_DIR}/hiveUtil.h
)

set(main_src
    main.c
    hiveConfig.c
    hiveUtil.c
)

add_library(hive SHARED ${main_inc} ${main_src} $<TARGET_OBJECTS:gas> $<TARGET_OBJECTS:network> $<TARGET_OBJECTS:introspection> $<TARGET_OBJECTS:runtime> $<TARGET_OBJECTS:system> $<TARGET_OBJECTS:common>)
install(FILES ${main_inc} DESTINATION include)
install(TARGETS hive LIBRARY DESTINATION lib)