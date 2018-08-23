add_subdirectory(common)
add_subdirectory(gas)
add_subdirectory(introspection)
add_subdirectory(network)
add_subdirectory(system)
add_subdirectory(runtime)

set(main_inc
    ${CORE_INCLUDE_DIR}/artsRT.h
    ${CORE_INCLUDE_DIR}/arts.h
    ${CORE_INCLUDE_DIR}/artsGlobals.h
    ${CORE_INCLUDE_DIR}/artsConfig.h
)

set(main_src
    main.c
    artsConfig.c
    artsUtil.c
)

add_library(arts STATIC ${main_inc} ${main_src} $<TARGET_OBJECTS:gas> $<TARGET_OBJECTS:network> $<TARGET_OBJECTS:introspection> $<TARGET_OBJECTS:runtime> $<TARGET_OBJECTS:system> $<TARGET_OBJECTS:common>)
install(FILES ${main_inc} DESTINATION include)
install(TARGETS arts ARCHIVE DESTINATION lib)