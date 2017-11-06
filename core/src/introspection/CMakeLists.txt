set(introspection_src
    hiveCounter.c
    hiveIntrospection.c
)

add_library(introspection OBJECT ${introspection_src})
