set(introspection_inc
    ${CORE_INCLUDE_DIR}/hiveCounter.h
    ${CORE_INCLUDE_DIR}/hiveIntrospection.h
)

set(introspection_src
    hiveCounter.c
    hiveIntrospection.c
)

add_library(introspection OBJECT ${introspection_inc} ${introspection_src})
install(FILES ${introspection_inc} DESTINATION include)