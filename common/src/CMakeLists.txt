if(DEQUE MATCHES "chase_lev")
    set(common_lib_src ${common_lib_src} chaseLevDequeSimple.c)
elseif( DEQUE MATCHES "priority" )
    set(common_lib_src ${common_lib_src} chaseLevDequePriority.c)
else()
    message( FATAL_ERROR "Unsupported runtime deque chosen.")
endif()

set(common_lib_src
    ${common_lib_src}
    hashLock.c
    hiveTimer.c
    table.c
    linkList.c
    hiveArrayList.c
    hiveQueue.c
)

add_library(common OBJECT ${common_lib_src})
