set(introspection_inc
    ${CORE_INCLUDE_DIR}/artsCounter.h
    ${CORE_INCLUDE_DIR}/artsIntrospection.h
)

set(introspection_src
    artsCounter.c
    artsIntrospection.c
)

add_library(introspection OBJECT ${introspection_inc} ${introspection_src})
install(FILES ${introspection_inc} DESTINATION include)